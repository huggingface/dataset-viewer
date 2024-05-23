# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from typing import Union

import polars as pl
import pytest

from libcommon.statistics import (
    NUM_BINS,
    BoolColumn,
    ClassLabelColumn,
    ColumnType,
    FloatColumn,
    IntColumn,
    ListColumn,
    generate_bins,
)

from .statistics_dataset import statistics_dataset
from .statistics_utils import (
    count_expected_statistics_for_bool_column,
    count_expected_statistics_for_categorical_column,
    count_expected_statistics_for_list_column,
    count_expected_statistics_for_numerical_column,
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


@pytest.mark.parametrize(
    "column_name",
    [
        "float__column",
        "float__nan_column",
        "float__all_nan_column",
        "float__negative_column",
        "float__cross_zero_column",
        "float__large_values_column",
        "float__only_one_value_column",
        "float__only_one_value_nan_column",
    ],
)
def test_float_statistics(
    column_name: str,
) -> None:
    data = statistics_dataset.to_pandas()
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
        "int__nan_column",
        "int__all_nan_column",
        "int__negative_column",
        "int__cross_zero_column",
        "int__large_values_column",
        "int__only_one_value_column",
        "int__only_one_value_nan_column",
    ],
)
def test_int_statistics(
    column_name: str,
) -> None:
    data = statistics_dataset.to_pandas()
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


# @pytest.mark.parametrize(
#     "column_name",
#     [
#         "string_text__column",
#         "string_text__nan_column",
#         "string_text__large_string_column",
#         "string_text__large_string_nan_column",
#         "string_label__column",
#         "string_label__nan_column",
#         "string_label__all_nan_column",
#     ],
# )
# def test_string_statistics(
#     column_name: str,
#     descriptive_statistics_expected: dict,  # type: ignore
#     descriptive_statistics_string_text_expected: dict,  # type: ignore
#     datasets: Mapping[str, Dataset],
# ) -> None:
#     if column_name.startswith("string_text__"):
#         expected = descriptive_statistics_string_text_expected["statistics"][column_name]["column_statistics"]
#         data = datasets["descriptive_statistics_string_text"].to_dict()
#     else:
#         expected = descriptive_statistics_expected["statistics"][column_name]["column_statistics"]
#         data = datasets["descriptive_statistics"].to_dict()
#     computed = StringColumn.compute_statistics(
#         data=pl.from_dict(data),
#         column_name=column_name,
#         n_samples=len(data[column_name]),
#     )
#     if column_name.startswith("string_text__"):
#         expected_hist, computed_hist = expected.pop("histogram"), computed.pop("histogram")
#         assert expected_hist["hist"] == computed_hist["hist"]
#         assert expected_hist["bin_edges"] == pytest.approx(computed_hist["bin_edges"])
#         assert expected == pytest.approx(computed)
#     else:
#         assert expected == computed


@pytest.mark.parametrize(
    "column_name",
    [
        "class_label__column",
        "class_label__nan_column",
        "class_label__all_nan_column",
        "class_label__less_classes_column",
        "class_label__string_column",
        "class_label__string_nan_column",
        "class_label__string_all_nan_column",
    ],
)
def test_class_label_statistics(
    column_name: str,
) -> None:
    data = statistics_dataset.to_pandas()
    class_label_feature = statistics_dataset.features[column_name]
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
        "bool__column",
        "bool__nan_column",
        "bool__all_nan_column",
    ],
)
def test_bool_statistics(
    column_name: str,
) -> None:
    data = statistics_dataset.to_pandas()
    expected = count_expected_statistics_for_bool_column(data[column_name])
    computed = BoolColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    assert computed == expected


@pytest.mark.parametrize(
    "column_name",
    [
        "list__int_column",
        "list__int_nan_column",
        "list__int_all_nan_column",
        "list__string_column",
        "list__string_nan_column",
        "list__string_all_nan_column",
        "list__dict_column",
        "list__dict_nan_column",
        "list__dict_all_nan_column",
        "list__sequence_int_column",
        "list__sequence_int_nan_column",
        "list__sequence_int_all_nan_column",
        "list__sequence_class_label_column",
        "list__sequence_class_label_nan_column",
        "list__sequence_class_label_all_nan_column",
        "list__sequence_of_sequence_bool_column",
        "list__sequence_of_sequence_bool_nan_column",
        "list__sequence_of_sequence_bool_all_nan_column",
        "list__sequence_of_sequence_dict_column",
        "list__sequence_of_sequence_dict_nan_column",
        "list__sequence_of_sequence_dict_all_nan_column",
        "list__sequence_of_list_dict_column",
        "list__sequence_of_list_dict_nan_column",
        "list__sequence_of_list_dict_all_nan_column",
    ],
)
def test_list_statistics(
    column_name: str,
) -> None:
    data = statistics_dataset.to_pandas()
    expected = count_expected_statistics_for_list_column(data[column_name])
    computed = ListColumn.compute_statistics(
        data=pl.from_pandas(data),
        column_name=column_name,
        n_samples=len(data[column_name]),
    )
    assert computed == expected
