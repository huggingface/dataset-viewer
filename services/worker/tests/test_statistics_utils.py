# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
import datetime
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import ClassLabel, Dataset
from datasets.table import embed_table_storage

from worker.statistics_utils import (
    DECIMALS,
    MAX_NUM_STRING_LABELS,
    MAX_PROPORTION_STRING_LABELS,
    NO_LABEL_VALUE,
    NUM_BINS,
    AudioColumn,
    BoolColumn,
    ClassLabelColumn,
    ColumnType,
    DatetimeColumn,
    FloatColumn,
    ImageColumn,
    IntColumn,
    ListColumn,
    StringColumn,
    generate_bins,
)


@pytest.mark.parametrize(
    "min_value,max_value,column_type,expected_bins",
    [
        (0, 1, ColumnType.INT, [0, 1, 1]),
        (0, 12, ColumnType.INT, [0, 2, 4, 6, 8, 10, 12, 12]),
        (-10, 15, ColumnType.INT, [-10, -7, -4, -1, 2, 5, 8, 11, 14, 15]),
        (0, 9, ColumnType.INT, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
        (0, 10, ColumnType.INT, [0, 2, 4, 6, 8, 10, 10]),
        (0.0, 10.0, ColumnType.FLOAT, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        (0.0, 0.1, ColumnType.FLOAT, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
        (0, 0, ColumnType.INT, [0, 0]),
        (0.0, 0.0, ColumnType.INT, [0.0, 0.0]),
        (-0.5, 0.5, ColumnType.FLOAT, [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        (-100.0, 100.0, ColumnType.FLOAT, [-100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0]),
    ],
)
def test_generate_bins(
    min_value: Union[int, float],
    max_value: Union[int, float],
    column_type: ColumnType,
    expected_bins: list[Union[int, float]],
) -> None:
    bins = generate_bins(
        min_value=min_value, max_value=max_value, column_name="dummy", column_type=column_type, n_bins=NUM_BINS
    )
    assert 2 <= len(bins) <= NUM_BINS + 1
    if column_type is column_type.FLOAT:
        assert pytest.approx(bins) == expected_bins
    else:
        assert bins == expected_bins


def count_expected_statistics_for_numerical_column(
    column: pd.Series,  # type: ignore
    dtype: ColumnType,
) -> dict:  # type: ignore
    minimum, maximum, mean, median, std = (
        column.min(),
        column.max(),
        column.mean(),
        column.median(),
        column.std(),
    )
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    if nan_count == n_samples:
        return {
            "nan_count": n_samples,
            "nan_proportion": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "histogram": None,
        }
    if dtype is ColumnType.FLOAT:
        if minimum == maximum:
            hist, bin_edges = np.array([column[~column.isna()].count()]), np.array([minimum, maximum])
        else:
            hist, bin_edges = np.histogram(column[~column.isna()], bins=NUM_BINS)
        bin_edges = bin_edges.astype(float).round(DECIMALS).tolist()
    else:
        bins = generate_bins(minimum, maximum, column_name="dummy", column_type=dtype, n_bins=NUM_BINS)
        hist, bin_edges = np.histogram(column[~column.isna()], bins)
        bin_edges = bin_edges.astype(int).tolist()
    hist = hist.astype(int).tolist()
    if hist and sum(hist) != column.shape[0] - nan_count:
        raise ValueError("incorrect hist")
    if dtype is ColumnType.FLOAT:
        minimum = minimum.astype(float).round(DECIMALS).item()
        maximum = maximum.astype(float).round(DECIMALS).item()
        mean = mean.astype(float).round(DECIMALS).item()  # type: ignore
        median = median.astype(float).round(DECIMALS).item()  # type: ignore
        std = std.astype(float).round(DECIMALS).item()  # type: ignore
    else:
        mean, median, std = list(np.round([mean, median, std], DECIMALS))
    return {
        "nan_count": nan_count,
        "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "median": median,
        "std": std,
        "histogram": {
            "hist": hist,
            "bin_edges": bin_edges,
        },
    }


def count_expected_statistics_for_list_column(column: pd.Series) -> dict:  # type: ignore
    if column.isnull().all():
        lengths_column = pd.Series([None] * column.shape[0])
        return count_expected_statistics_for_numerical_column(lengths_column, dtype=ColumnType.INT)
    column_without_na = column.dropna()
    first_sample = column_without_na.iloc[0]
    if isinstance(first_sample, dict):  # sequence is dict of lists
        lengths_column = column.map(lambda x: len(next(iter(x.values()))) if x is not None else None)
    else:
        lengths_column = column.map(lambda x: len(x) if x is not None else None)
    return count_expected_statistics_for_numerical_column(lengths_column, dtype=ColumnType.INT)


def count_expected_statistics_for_categorical_column(
    column: pd.Series,  # type: ignore
    class_label_feature: ClassLabel,
) -> dict:  # type: ignore
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    value_counts = column.value_counts(dropna=True).to_dict()
    no_label_count = int(value_counts.pop(NO_LABEL_VALUE, 0))
    num_classes = len(class_label_feature.names)
    frequencies = {
        class_label_feature.int2str(int(class_id)): value_counts.get(class_id, 0) for class_id in range(num_classes)
    }
    return {
        "nan_count": nan_count,
        "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
        "no_label_count": no_label_count,
        "no_label_proportion": np.round(no_label_count / n_samples, DECIMALS).item() if no_label_count else 0.0,
        "n_unique": num_classes,
        "frequencies": frequencies,
    }


def count_expected_statistics_for_string_column(column: pd.Series) -> dict:  # type: ignore
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    value_counts = column.value_counts(dropna=True).to_dict()
    n_unique = len(value_counts)
    if (
        n_unique / n_samples <= MAX_PROPORTION_STRING_LABELS
        and n_unique <= MAX_NUM_STRING_LABELS
        or n_unique <= NUM_BINS
    ):
        return {
            "nan_count": nan_count,
            "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
            "no_label_count": 0,
            "no_label_proportion": 0.0,
            "n_unique": n_unique,
            "frequencies": value_counts,
        }

    lengths_column = column.map(lambda x: len(x) if x is not None else None)
    return count_expected_statistics_for_numerical_column(lengths_column, dtype=ColumnType.INT)


def count_expected_statistics_for_bool_column(column: pd.Series) -> dict:  # type: ignore
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    value_counts = column.value_counts(dropna=True).to_dict()
    return {
        "nan_count": nan_count,
        "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
        "frequencies": {str(key): freq for key, freq in value_counts.items()},
    }


@pytest.mark.parametrize(
    "column_name",
    [
        "float__column",
        "float__null_column",
        "float__nan_column",
        "float__all_null_column",
        "float__all_nan_column",
        "float__negative_column",
        "float__cross_zero_column",
        "float__large_values_column",
        "float__only_one_value_column",
        "float__only_one_value_null_column",
    ],
)
def test_float_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    data = datasets["descriptive_statistics"].to_pandas()
    expected = count_expected_statistics_for_numerical_column(data[column_name], dtype=ColumnType.FLOAT)
    computed = FloatColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    expected_hist, computed_hist = expected.pop("histogram"), computed.pop("histogram")
    if computed_hist:
        assert computed_hist["hist"] == expected_hist["hist"]
        assert pytest.approx(computed_hist["bin_edges"]) == expected_hist["bin_edges"]
    assert pytest.approx(computed) == expected
    assert computed["nan_count"] == expected["nan_count"]


@pytest.mark.parametrize(
    "column_name",
    [
        "int__column",
        "int__null_column",
        "int__all_null_column",
        "int__negative_column",
        "int__cross_zero_column",
        "int__large_values_column",
        "int__only_one_value_column",
        "int__only_one_value_null_column",
    ],
)
def test_int_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    data = datasets["descriptive_statistics"].to_pandas()
    expected = count_expected_statistics_for_numerical_column(data[column_name], dtype=ColumnType.INT)
    computed = IntColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    expected_hist, computed_hist = expected.pop("histogram"), computed.pop("histogram")
    if computed_hist:
        assert computed_hist["hist"] == expected_hist["hist"]
        assert pytest.approx(computed_hist["bin_edges"]) == expected_hist["bin_edges"]
    assert pytest.approx(computed) == expected
    assert computed["nan_count"] == expected["nan_count"]
    assert computed["min"] == expected["min"]
    assert computed["max"] == expected["max"]


@pytest.mark.parametrize(
    "column_name",
    [
        "string_text__column",
        "string_text__null_column",
        "string_text__large_string_column",
        "string_text__large_string_null_column",
        "string_label__column",
        "string_label__null_column",
        "string_label__all_null_column",
    ],
)
def test_string_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    if column_name.startswith("string_text__"):
        data = datasets["descriptive_statistics_string_text"].to_pandas()
    else:
        data = datasets["descriptive_statistics"].to_pandas()
    expected = count_expected_statistics_for_string_column(data[column_name])
    computed = StringColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    if column_name.startswith("string_text__"):
        expected_hist, computed_hist = expected.pop("histogram"), computed.pop("histogram")
        assert expected_hist["hist"] == computed_hist["hist"]
        assert expected_hist["bin_edges"] == pytest.approx(computed_hist["bin_edges"])
        assert expected == pytest.approx(computed)
    else:
        assert expected == computed


@pytest.mark.parametrize(
    "column_name",
    [
        "class_label__column",
        "class_label__null_column",
        "class_label__all_null_column",
        "class_label__less_classes_column",
        "class_label__string_column",
        "class_label__string_null_column",
        "class_label__string_all_null_column",
    ],
)
def test_class_label_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    data = datasets["descriptive_statistics"].to_pandas()
    class_label_feature = datasets["descriptive_statistics"].features[column_name]
    expected = count_expected_statistics_for_categorical_column(data[column_name], class_label_feature)
    computed = ClassLabelColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
        feature_dict={"_type": "ClassLabel", "names": class_label_feature.names},
    )
    assert expected == computed


@pytest.mark.parametrize(
    "column_name",
    [
        "list__int_column",
        "list__int_null_column",
        "list__int_all_null_column",
        "list__string_column",
        "list__string_null_column",
        "list__string_all_null_column",
        "list__dict_column",
        "list__dict_null_column",
        "list__dict_all_null_column",
        "list__sequence_int_column",
        "list__sequence_int_null_column",
        "list__sequence_int_all_null_column",
        "list__sequence_class_label_column",
        "list__sequence_class_label_null_column",
        "list__sequence_class_label_all_null_column",
        "list__sequence_of_sequence_bool_column",
        "list__sequence_of_sequence_bool_null_column",
        "list__sequence_of_sequence_bool_all_null_column",
        "list__sequence_of_sequence_dict_column",
        "list__sequence_of_sequence_dict_null_column",
        "list__sequence_of_sequence_dict_all_null_column",
        "list__sequence_of_list_dict_column",
        "list__sequence_of_list_dict_null_column",
        "list__sequence_of_list_dict_all_null_column",
    ],
)
def test_list_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    data = datasets["descriptive_statistics"].to_pandas()
    expected = count_expected_statistics_for_list_column(data[column_name])
    computed = ListColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    assert computed == expected


@pytest.mark.parametrize(
    "column_name",
    [
        "bool__column",
        "bool__null_column",
        "bool__all_null_column",
    ],
)
def test_bool_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    data = datasets["descriptive_statistics"].to_pandas()
    expected = count_expected_statistics_for_bool_column(data[column_name])
    computed = BoolColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    assert computed == expected


@pytest.mark.parametrize(
    "column_name,audio_durations",
    [
        ("audio", [1.0, 2.0, 3.0, 4.0]),  # datasets consists of 4 audio files of 1, 2, 3, 4 seconds lengths
        ("audio_null", [1.0, None, 3.0, None]),  # take first and third audio file for this testcase
        ("audio_all_null", [None, None, None, None]),
    ],
)
def test_audio_statistics(
    column_name: str,
    audio_durations: Optional[list[Optional[float]]],
    datasets: Mapping[str, Dataset],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    expected = count_expected_statistics_for_numerical_column(
        column=pd.Series(audio_durations), dtype=ColumnType.FLOAT
    )
    parquet_directory = tmp_path_factory.mktemp("data")
    parquet_filename = parquet_directory / "data.parquet"
    dataset_table = datasets["audio_statistics"].data
    dataset_table_embedded = embed_table_storage(dataset_table)  # store audio as bytes instead of paths to files
    pq.write_table(dataset_table_embedded, parquet_filename)
    computed = AudioColumn.compute_statistics(
        parquet_directory=parquet_directory,
        column_name=column_name,
        n_samples=4,
    )
    assert computed == expected

    # write samples as just bytes, not as struct {"bytes": b"", "path": ""}, to check that this format works too
    audios = datasets["audio_statistics"][column_name][:]
    pa_table_bytes = pa.Table.from_pydict(
        {column_name: [open(audio["path"], "rb").read() if audio else None for audio in audios]}
    )
    pq.write_table(pa_table_bytes, parquet_filename)
    computed = AudioColumn.compute_statistics(
        parquet_directory=parquet_directory,
        column_name=column_name,
        n_samples=4,
    )
    assert computed == expected


@pytest.mark.parametrize(
    "column_name,image_widths",
    [
        ("image", [640, 1440, 520, 1240]),  # datasets consists of 4 image files
        ("image_null", [640, None, 520, None]),  # take first and third image file for this testcase
        ("image_all_null", [None, None, None, None]),
    ],
)
def test_image_statistics(
    column_name: str,
    image_widths: Optional[list[Optional[float]]],
    datasets: Mapping[str, Dataset],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    expected = count_expected_statistics_for_numerical_column(column=pd.Series(image_widths), dtype=ColumnType.INT)
    parquet_directory = tmp_path_factory.mktemp("data")
    parquet_filename = parquet_directory / "data.parquet"
    dataset_table = datasets["image_statistics"].data
    dataset_table_embedded = embed_table_storage(dataset_table)  # store image as bytes instead of paths to files
    pq.write_table(dataset_table_embedded, parquet_filename)
    computed = ImageColumn.compute_statistics(
        parquet_directory=parquet_directory,
        column_name=column_name,
        n_samples=4,
    )
    assert computed == expected

    # write samples as just bytes, not as struct {"bytes": b"", "path": ""}, to check that this format works too
    images = datasets["image_statistics"][column_name][:]
    pa_table_bytes = pa.Table.from_pydict(
        {column_name: [open(image["path"], "rb").read() if image else None for image in images]}
    )
    pq.write_table(pa_table_bytes, parquet_filename)
    computed = ImageColumn.compute_statistics(
        parquet_directory=parquet_directory,
        column_name=column_name,
        n_samples=4,
    )
    assert computed == expected


def count_expected_statistics_for_datetime(column: pd.Series, column_name: str) -> dict:  # type: ignore
    n_samples = column.shape[0]
    nan_count = column.isna().sum()
    if nan_count == n_samples:
        return {
            "nan_count": n_samples,
            "nan_proportion": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "histogram": None,
        }

    # hardcode expected values
    minv = "2024-01-01 00:00:00"
    maxv = "2024-01-11 00:00:00"
    mean = "2024-01-06 00:00:00"
    median = "2024-01-06 00:00:00"
    bin_edges = [
        "2024-01-01 00:00:00",
        "2024-01-02 00:00:01",
        "2024-01-03 00:00:02",
        "2024-01-04 00:00:03",
        "2024-01-05 00:00:04",
        "2024-01-06 00:00:05",
        "2024-01-07 00:00:06",
        "2024-01-08 00:00:07",
        "2024-01-09 00:00:08",
        "2024-01-10 00:00:09",
        "2024-01-11 00:00:00",
    ]

    # compute std
    seconds_in_day = 24 * 60 * 60
    if column_name == "datetime":
        timedeltas = pd.Series(range(0, 11 * seconds_in_day, seconds_in_day))
        hist = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif column_name == "datetime_null":
        timedeltas = pd.Series(range(0, 6 * 2 * seconds_in_day, 2 * seconds_in_day))  # take every second day
        hist = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    else:
        raise ValueError("Incorrect column")

    std = timedeltas.std()
    std_str = str(datetime.timedelta(seconds=std))

    return {
        "nan_count": nan_count,
        "nan_proportion": np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0,
        "min": minv,
        "max": maxv,
        "mean": mean,
        "median": median,
        "std": std_str,
        "histogram": {
            "hist": hist,
            "bin_edges": bin_edges,
        },
    }


@pytest.mark.parametrize(
    "column_name",
    ["datetime", "datetime_null", "datetime_all_null"],
)
def test_datetime_statistics(
    column_name: str,
    datasets: Mapping[str, Dataset],
) -> None:
    data = datasets["datetime_statistics"].to_pandas()
    expected = count_expected_statistics_for_datetime(data[column_name], column_name)
    computed = DatetimeColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    computed_std, expected_std = computed.pop("std"), expected.pop("std")
    if computed_std:
        assert computed_std.split(".")[0] == expected_std.split(".")[0]  # check with precision up to seconds
    else:
        assert computed_std == expected_std
    assert computed == expected
