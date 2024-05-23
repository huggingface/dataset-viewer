import numpy as np
import pandas as pd
from datasets import ClassLabel

from libcommon.statistics import (
    DECIMALS,
    MAX_NUM_STRING_LABELS,
    MAX_PROPORTION_STRING_LABELS,
    NO_LABEL_VALUE,
    NUM_BINS,
    ColumnType,
    generate_bins,
)


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
