# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
import datetime
import enum
import io
import logging
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict, Union

import librosa
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from datasets import Features
from libcommon.exceptions import (
    StatisticsComputationError,
)
from PIL import Image
from tqdm.contrib.concurrent import thread_map

DECIMALS = 5
NUM_BINS = 10

# the maximum proportion of unique values in a string column so that it is considered to be a category,
# otherwise it's treated as a string
# for example, 21 unique values in 100 samples -> strings
# 200 unique values in 1000 samples -> category
# 201 unique values in 1000 samples -> string
MAX_PROPORTION_STRING_LABELS = 0.2
# if there are more than MAX_NUM_STRING_LABELS unique strings, consider column to be a string
# even if proportion of unique values is lower than MAX_PROPORTION_STRING_LABELS
MAX_NUM_STRING_LABELS = 1000

# datasets.ClassLabel feature uses -1 to encode `no label` value
NO_LABEL_VALUE = -1


INTEGER_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
FLOAT_DTYPES = ["float16", "float32", "float64"]
NUMERICAL_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES
STRING_DTYPES = ["string", "large_string"]


class ColumnType(str, enum.Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    LIST = "list"
    CLASS_LABEL = "class_label"
    STRING_LABEL = "string_label"
    STRING_TEXT = "string_text"
    AUDIO = "audio"
    IMAGE = "image"
    DATETIME = "datetime"


class Histogram(TypedDict):
    hist: list[int]
    bin_edges: list[Union[int, float]]


class DatetimeHistogram(TypedDict):
    hist: list[int]
    bin_edges: list[str]  # edges are string representations of dates


class NumericalStatisticsItem(TypedDict):
    nan_count: int
    nan_proportion: float
    min: Optional[Union[int, float]]  # might be None in very rare cases when the whole column is only None values
    max: Optional[Union[int, float]]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    histogram: Optional[Histogram]


class DatetimeStatisticsItem(TypedDict):
    nan_count: int
    nan_proportion: float
    min: Optional[str]  # might be None in very rare cases when the whole column is only None values
    max: Optional[str]
    mean: Optional[str]
    median: Optional[str]
    std: Optional[str]  # string representation of timedelta
    histogram: Optional[DatetimeHistogram]


class CategoricalStatisticsItem(TypedDict):
    nan_count: int
    nan_proportion: float
    no_label_count: int
    no_label_proportion: float
    n_unique: int
    frequencies: dict[str, int]


class BoolStatisticsItem(TypedDict):
    nan_count: int
    nan_proportion: float
    frequencies: dict[str, int]


SupportedStatistics = Union[
    NumericalStatisticsItem, CategoricalStatisticsItem, BoolStatisticsItem, DatetimeStatisticsItem
]


class StatisticsPerColumnItem(TypedDict):
    column_name: str
    column_type: ColumnType
    column_statistics: SupportedStatistics


def generate_bins(
    min_value: Union[int, float],
    max_value: Union[int, float],
    column_type: ColumnType,
    n_bins: int,
    column_name: Optional[str] = None,
) -> list[Union[int, float]]:
    """
    Compute bin edges for float and int. Note that for int data returned number of edges (= number of bins)
    might be *less* than provided `n_bins` + 1 since (`max_value` - `min_value` + 1) might be not divisible by `n_bins`,
    therefore, we adjust the number of bins so that bin_size is always a natural number.
    bin_size for int data is calculated as np.ceil((max_value - min_value + 1) / n_bins)
    For float numbers, length of returned bin edges list is always equal to `n_bins` except for the cases
    when min = max (only one value observed in data). In this case, bin edges are [min, max].

    Raises:
        [~`libcommon.exceptions.StatisticsComputationError`]:
            If there was some unexpected behaviour during statistics computation.

    Returns:
        `list[Union[int, float]]`: List of bin edges of lengths <= n_bins + 1 and >= 2.
    """
    if column_type is ColumnType.FLOAT:
        if min_value == max_value:
            bin_edges = [min_value]
        else:
            bin_size = (max_value - min_value) / n_bins
            bin_edges = np.arange(min_value, max_value, bin_size).astype(float).tolist()
            if len(bin_edges) != n_bins:
                raise StatisticsComputationError(
                    f"Incorrect number of bins generated for {column_name=}, expected {n_bins}, got {len(bin_edges)}."
                )
    elif column_type is ColumnType.INT:
        bin_size = np.ceil((max_value - min_value + 1) / n_bins)
        bin_edges = np.arange(min_value, max_value + 1, bin_size).astype(int).tolist()
        if len(bin_edges) > n_bins:
            raise StatisticsComputationError(
                f"Incorrect number of bins generated for {column_name=}, expected {n_bins}, got {len(bin_edges)}."
            )
    else:
        raise ValueError(f"Incorrect column type of {column_name=}: {column_type}. ")
    return bin_edges + [max_value]


def compute_histogram(
    df: pl.dataframe.frame.DataFrame,
    column_name: str,
    column_type: ColumnType,
    min_value: Union[int, float],
    max_value: Union[int, float],
    n_bins: int,
    n_samples: int,
) -> Histogram:
    """
    Compute histogram over numerical (int and float) data using `polars`.
    `polars` histogram implementation uses left half-open intervals in bins, while more standard approach
    (implemented in `numpy`, for example) would be to use right half-open intervals
    (except for the last bin, which is closed to include the maximum value).
    In order to be aligned with this, this function first multiplies all values in column and bin edges by -1,
    computes histogram with `polars` using these inverted numbers, and then reverses everything back.
    """

    logging.debug(f"Compute histogram for {column_name=}")
    bin_edges = generate_bins(
        min_value=min_value, max_value=max_value, column_name=column_name, column_type=column_type, n_bins=n_bins
    )
    if len(bin_edges) == 2:  # possible if min == max (=data has only one value)
        if bin_edges[0] != bin_edges[1]:
            raise StatisticsComputationError(
                f"Got unexpected result during histogram computation for {column_name=}, {column_type=}: "
                f" len({bin_edges=}) is 2 but {bin_edges[0]=} != {bin_edges[1]=}. "
            )
        hist = [int(df[column_name].is_not_null().sum())]
    elif len(bin_edges) > 2:
        bins_edges_reverted = [-1 * b for b in bin_edges[::-1]]
        hist_df_reverted = df.with_columns(pl.col(column_name).mul(-1).alias("reverse"))["reverse"].hist(
            bins=bins_edges_reverted
        )
        hist_reverted = hist_df_reverted["count"].cast(int).to_list()
        hist = hist_reverted[::-1]
        hist = [hist[0] + hist[1]] + hist[2:-2] + [hist[-2] + hist[-1]]
    else:
        raise StatisticsComputationError(
            f"Got unexpected result during histogram computation for {column_name=}, {column_type=}: "
            f" unexpected {bin_edges=}"
        )
    logging.debug(f"{hist=} {bin_edges=}")

    if len(hist) != len(bin_edges) - 1:
        raise StatisticsComputationError(
            f"Got unexpected result during histogram computation for {column_name=}, {column_type=}: "
            f" number of bins in hist counts and bin edges don't match {hist=}, {bin_edges=}"
        )
    if sum(hist) != n_samples:
        raise StatisticsComputationError(
            f"Got unexpected result during histogram computation for {column_name=}, {column_type=}: "
            f" hist counts sum and number of non-null samples don't match, histogram sum={sum(hist)}, {n_samples=}"
        )

    return Histogram(
        hist=hist, bin_edges=np.round(bin_edges, DECIMALS).tolist() if column_type is column_type.FLOAT else bin_edges
    )


def min_max_mean_median_std(data: pl.DataFrame, column_name: str) -> tuple[float, float, float, float, float]:
    """
    Compute minimum, maximum, median, standard deviation, number of nan samples and their proportion in column data.
    """
    col_stats = dict(
        min=pl.all().min(),
        max=pl.all().max(),
        mean=pl.all().mean(),
        median=pl.all().median(),
        std=pl.all().std(),
    )
    stats_names = pl.Series(col_stats.keys())
    stats_expressions = [pl.struct(stat) for stat in col_stats.values()]
    stats = (
        data.select(pl.col(column_name))
        .select(name=stats_names, stats=pl.concat_list(stats_expressions).flatten())
        .unnest("stats")
    )
    minimum, maximum, mean, median, std = stats[column_name].to_list()
    if any(statistic is None for statistic in [minimum, maximum, mean, median, std]):
        # this should be possible only if all values are none
        if not all(statistic is None for statistic in [minimum, maximum, mean, median, std]):
            raise StatisticsComputationError(
                f"Unexpected result for {column_name=}: "
                f"Some measures among {minimum=}, {maximum=}, {mean=}, {median=}, {std=} are None but not all of them. "
            )
        return minimum, maximum, mean, median, std

    minimum, maximum, mean, median, std = np.round([minimum, maximum, mean, median, std], DECIMALS).tolist()

    return minimum, maximum, mean, median, std


def value_counts(data: pl.DataFrame, column_name: str) -> dict[Any, Any]:
    """Compute counts of distinct values in a column of a dataframe."""

    return dict(data[column_name].value_counts().rows())


def nan_count_proportion(data: pl.DataFrame, column_name: str, n_samples: int) -> tuple[int, float]:
    nan_count = data[column_name].null_count()
    nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0
    return nan_count, nan_proportion


def all_nan_statistics_item(n_samples: int) -> NumericalStatisticsItem:
    return NumericalStatisticsItem(
        nan_count=n_samples,
        nan_proportion=1.0,
        min=None,
        max=None,
        mean=None,
        median=None,
        std=None,
        histogram=None,
    )


class Column:
    """Abstract class to compute stats for columns of all supported data types."""

    # Optional column class that might be used inside ._compute_statistics() if computation should be performed
    # over some transformed values. For example, for StringColumn.transform_column is IntColumn
    # because stats are calculated over string lengths which are integers.
    transform_column: Optional[type["Column"]] = None

    def __init__(
        self,
        feature_name: str,
        n_samples: int,
    ):
        self.name = feature_name
        self.n_samples = n_samples

    @classmethod
    def compute_transformed_data(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @classmethod
    def _compute_statistics(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> SupportedStatistics:
        raise NotImplementedError

    @classmethod
    def compute_statistics(
        cls,
        *args: Any,
        column_name: str,
        **kwargs: Any,
    ) -> Any:
        try:
            logging.info(f"Compute statistics for {cls.__name__} {column_name}. ")
            return cls._compute_statistics(*args, column_name=column_name, **kwargs)
        except Exception as error:
            raise StatisticsComputationError(f"Error for {cls.__name__}={column_name}: {error=}", error)

    def compute_and_prepare_response(self, *args: Any, **kwargs: Any) -> StatisticsPerColumnItem:
        raise NotImplementedError


class ClassLabelColumn(Column):
    def __init__(self, *args: Any, feature_dict: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.feature_dict = feature_dict

    @classmethod
    def _compute_statistics(
        cls, data: pl.DataFrame, column_name: str, n_samples: int, feature_dict: dict[str, Any]
    ) -> CategoricalStatisticsItem:
        datasets_feature = Features.from_dict({column_name: feature_dict})[column_name]
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples)

        ids2counts: dict[int, int] = value_counts(data, column_name)
        no_label_count = ids2counts.pop(NO_LABEL_VALUE, 0)
        no_label_proportion = np.round(no_label_count / n_samples, DECIMALS).item() if no_label_count != 0 else 0.0

        num_classes = len(datasets_feature.names)
        labels2counts: dict[str, int] = {
            datasets_feature.int2str(cat_id): ids2counts.get(cat_id, 0) for cat_id in range(num_classes)
        }
        n_unique = data[column_name].n_unique()
        logging.debug(
            f"{nan_count=} {nan_proportion=} {no_label_count=} {no_label_proportion=}, {n_unique=} {labels2counts=}"
        )

        if n_unique > num_classes + int(no_label_count > 0) + int(nan_count > 0):
            raise StatisticsComputationError(
                f"Got unexpected result for ClassLabel {column_name=}: "
                f" number of unique values is greater than provided by feature metadata. "
                f" {n_unique=}, {datasets_feature=}, {no_label_count=}, {nan_count=}. "
            )

        return CategoricalStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            no_label_count=no_label_count,
            no_label_proportion=no_label_proportion,
            n_unique=num_classes,
            frequencies=labels2counts,
        )

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(
            data, column_name=self.name, n_samples=self.n_samples, feature_dict=self.feature_dict
        )
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.CLASS_LABEL,
            column_statistics=stats,
        )


class FloatColumn(Column):
    @classmethod
    def _compute_statistics(
        cls,
        data: pl.DataFrame,
        column_name: str,
        n_samples: int,
    ) -> NumericalStatisticsItem:
        data = data.fill_nan(None)
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples)
        if nan_count == n_samples:  # all values are None
            return all_nan_statistics_item(n_samples)

        minimum, maximum, mean, median, std = min_max_mean_median_std(data, column_name)
        logging.debug(f"{minimum=}, {maximum=}, {mean=}, {median=}, {std=}, {nan_count=} {nan_proportion=}")

        hist = compute_histogram(
            data,
            column_name=column_name,
            column_type=ColumnType.FLOAT,
            min_value=minimum,
            max_value=maximum,
            n_bins=NUM_BINS,
            n_samples=n_samples - nan_count,
        )

        return NumericalStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            min=minimum,
            max=maximum,
            mean=mean,
            median=median,
            std=std,
            histogram=hist,
        )

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(data, column_name=self.name, n_samples=self.n_samples)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.FLOAT,
            column_statistics=stats,
        )


class IntColumn(Column):
    @classmethod
    def _compute_statistics(
        cls,
        data: pl.DataFrame,
        column_name: str,
        n_samples: int,
    ) -> NumericalStatisticsItem:
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples=n_samples)
        if nan_count == n_samples:
            return all_nan_statistics_item(n_samples)

        minimum, maximum, mean, median, std = min_max_mean_median_std(data, column_name)
        logging.debug(f"{minimum=}, {maximum=}, {mean=}, {median=}, {std=}, {nan_count=} {nan_proportion=}")

        minimum, maximum = int(minimum), int(maximum)
        hist = compute_histogram(
            data,
            column_name=column_name,
            column_type=ColumnType.INT,
            min_value=minimum,
            max_value=maximum,
            n_bins=NUM_BINS,
            n_samples=n_samples - nan_count,
        )

        return NumericalStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            min=minimum,
            max=maximum,
            mean=mean,
            median=median,
            std=std,
            histogram=hist,
        )

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(data, column_name=self.name, n_samples=self.n_samples)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.INT,
            column_statistics=stats,
        )


class StringColumn(Column):
    transform_column = IntColumn

    @staticmethod
    def is_class(n_unique: int, n_samples: int) -> bool:
        return (
            n_unique / n_samples <= MAX_PROPORTION_STRING_LABELS and n_unique <= MAX_NUM_STRING_LABELS
        ) or n_unique <= NUM_BINS

    @classmethod
    def compute_transformed_data(
        cls,
        data: pl.DataFrame,
        column_name: str,
        transformed_column_name: str,
    ) -> pl.DataFrame:
        return data.select(pl.col(column_name)).with_columns(
            pl.col(column_name).str.len_chars().alias(transformed_column_name)
        )

    @classmethod
    def _compute_statistics(
        cls,
        data: pl.DataFrame,
        column_name: str,
        n_samples: int,
    ) -> Union[CategoricalStatisticsItem, NumericalStatisticsItem]:
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples)
        n_unique = data[column_name].n_unique()
        if cls.is_class(n_unique, n_samples):
            labels2counts: dict[str, int] = value_counts(data, column_name) if nan_count != n_samples else {}
            logging.debug(f"{n_unique=} {nan_count=} {nan_proportion=} {labels2counts=}")
            # exclude counts of None values from frequencies if exist:
            labels2counts.pop(None, None)  # type: ignore
            return CategoricalStatisticsItem(
                nan_count=nan_count,
                nan_proportion=nan_proportion,
                no_label_count=0,
                no_label_proportion=0.0,
                n_unique=len(labels2counts),
                frequencies=labels2counts,
            )

        lengths_column_name = f"{column_name}_len"
        lengths_df = cls.compute_transformed_data(data, column_name, transformed_column_name=lengths_column_name)
        lengths_stats: NumericalStatisticsItem = cls.transform_column.compute_statistics(
            lengths_df, column_name=lengths_column_name, n_samples=n_samples
        )
        return lengths_stats

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(data, column_name=self.name, n_samples=self.n_samples)
        string_type = ColumnType.STRING_LABEL if "frequencies" in stats else ColumnType.STRING_TEXT
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=string_type,
            column_statistics=stats,
        )


class BoolColumn(Column):
    @classmethod
    def _compute_statistics(cls, data: pl.DataFrame, column_name: str, n_samples: int) -> BoolStatisticsItem:
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples)
        values2counts: dict[str, int] = value_counts(data, column_name)
        # exclude counts of None values from frequencies if exist:
        values2counts.pop(None, None)  # type: ignore
        logging.debug(f"{nan_count=} {nan_proportion=} {values2counts=}")
        return BoolStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            frequencies={str(key): freq for key, freq in sorted(values2counts.items())},
        )

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(data, column_name=self.name, n_samples=self.n_samples)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.BOOL,
            column_statistics=stats,
        )


class ListColumn(Column):
    transform_column = IntColumn

    @classmethod
    def compute_transformed_data(
        cls,
        data: pl.DataFrame,
        column_name: str,
        transformed_column_name: str,
    ) -> pl.DataFrame:
        return data.select(
            pl.col(column_name),
            pl.when(pl.col(column_name).is_not_null())
            .then(pl.col(column_name).list.len())
            .otherwise(pl.lit(None))  # polars counts len(null) in list type column as 0, while we want to keep null
            .alias(transformed_column_name),
        )

    @classmethod
    def _compute_statistics(
        cls,
        data: pl.DataFrame,
        column_name: str,
        n_samples: int,
    ) -> NumericalStatisticsItem:
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples)
        if nan_count == n_samples:
            return all_nan_statistics_item(n_samples)

        lengths_column_name = f"{column_name}_len"
        lengths_df = cls.compute_transformed_data(data, column_name, lengths_column_name)
        lengths_stats: NumericalStatisticsItem = cls.transform_column.compute_statistics(
            lengths_df, column_name=lengths_column_name, n_samples=n_samples
        )

        return NumericalStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            min=lengths_stats["min"],
            max=lengths_stats["max"],
            mean=lengths_stats["mean"],
            median=lengths_stats["median"],
            std=lengths_stats["std"],
            histogram=lengths_stats["histogram"],
        )

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(data, column_name=self.name, n_samples=self.n_samples)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.LIST,
            column_statistics=stats,
        )


class MediaColumn(Column):
    transform_column: type[Column]

    @classmethod
    def transform(cls, example: Optional[Union[bytes, dict[str, Any]]]) -> Any:
        """
        Function to use to transform the original values to further pass these transformed values to statistics
        computation. Used inside ._compute_statistics() method.
        """
        raise NotImplementedError

    @classmethod
    def compute_transformed_data(
        cls, parquet_directory: Path, column_name: str, transform_func: Callable[[Any], Any]
    ) -> list[Any]:
        parquet_files = list(parquet_directory.glob("*.parquet"))
        transformed_values = []
        for filename in parquet_files:
            shard_items = pq.read_table(filename, columns=[column_name]).to_pydict()[column_name]
            shard_transformed_values = thread_map(
                transform_func,
                shard_items,
                desc=f"Transforming values of {cls.__name__} {column_name} for {filename.name}",
                leave=False,
            )
            transformed_values.extend(shard_transformed_values)
        return transformed_values

    @classmethod
    def _compute_statistics(
        cls,
        parquet_directory: Path,
        column_name: str,
        n_samples: int,
    ) -> SupportedStatistics:
        transformed_values = cls.compute_transformed_data(parquet_directory, column_name, cls.transform)
        nan_count = sum(value is None for value in transformed_values)
        if nan_count == n_samples:
            return all_nan_statistics_item(n_samples)

        nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0
        transformed_df = pl.from_dict({column_name: transformed_values})
        transformed_stats: NumericalStatisticsItem = cls.transform_column.compute_statistics(
            data=transformed_df,
            column_name=column_name,
            n_samples=n_samples,
        )
        return NumericalStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            min=transformed_stats["min"],
            max=transformed_stats["max"],
            mean=transformed_stats["mean"],
            median=transformed_stats["median"],
            std=transformed_stats["std"],
            histogram=transformed_stats["histogram"],
        )

    @classmethod
    def get_column_type(cls) -> ColumnType:
        return ColumnType(cls.__name__.split("Column")[0].lower())

    def compute_and_prepare_response(self, parquet_directory: Path) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(
            parquet_directory=parquet_directory,
            column_name=self.name,
            n_samples=self.n_samples,
        )
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=self.get_column_type(),
            column_statistics=stats,
        )


class AudioColumn(MediaColumn):
    transform_column = FloatColumn

    @staticmethod
    def get_duration(example: Optional[Union[bytes, dict[str, Any]]]) -> Optional[float]:
        """Get audio durations"""
        if example is None:
            return None
        example_bytes = example["bytes"] if isinstance(example, dict) else example
        with io.BytesIO(example_bytes) as f:
            return librosa.get_duration(path=f)  # type: ignore   # expects PathLike but BytesIO also works

    @classmethod
    def transform(cls, example: Optional[Union[bytes, dict[str, Any]]]) -> Optional[float]:
        return cls.get_duration(example)


class ImageColumn(MediaColumn):
    transform_column = IntColumn

    @staticmethod
    def get_width(example: Optional[Union[bytes, dict[str, Any]]]) -> Optional[int]:
        """Get image widths."""
        image_shape = ImageColumn.get_shape(example)
        return image_shape[0]

    @staticmethod
    def get_shape(example: Optional[Union[bytes, dict[str, Any]]]) -> Union[tuple[None, None], tuple[int, int]]:
        """Get image widths and heights."""
        if example is None:
            return None, None
        example_bytes = example["bytes"] if isinstance(example, dict) else example
        with io.BytesIO(example_bytes) as f:
            image = Image.open(f)
            return image.size

    @classmethod
    def transform(cls, example: Optional[Union[bytes, dict[str, Any]]]) -> Optional[int]:
        return cls.get_width(example)


class DatetimeColumn(Column):
    transform_column = IntColumn

    @classmethod
    def compute_transformed_data(
        cls,
        data: pl.DataFrame,
        column_name: str,
        transformed_column_name: str,
        min_date: datetime.datetime,
    ) -> pl.DataFrame:
        return data.select((pl.col(column_name) - min_date).dt.total_seconds().alias(transformed_column_name))

    @staticmethod
    def shift_and_convert_to_string(base_date: datetime.datetime, seconds: Union[int, float]) -> str:
        return datetime_to_string(base_date + datetime.timedelta(seconds=seconds))

    @classmethod
    def _compute_statistics(
        cls,
        data: pl.DataFrame,
        column_name: str,
        n_samples: int,
    ) -> DatetimeStatisticsItem:
        nan_count, nan_proportion = nan_count_proportion(data, column_name, n_samples)
        if nan_count == n_samples:  # all values are None
            return DatetimeStatisticsItem(
                nan_count=n_samples,
                nan_proportion=1.0,
                min=None,
                max=None,
                mean=None,
                median=None,
                std=None,
                histogram=None,
            )

        min_date: datetime.datetime = data[column_name].min()  # type: ignore   # mypy infers type of datetime column .min() incorrectly
        timedelta_column_name = f"{column_name}_timedelta"
        # compute distribution of time passed from min date in **seconds**
        timedelta_df = cls.compute_transformed_data(data, column_name, timedelta_column_name, min_date)
        timedelta_stats: NumericalStatisticsItem = cls.transform_column.compute_statistics(
            timedelta_df,
            column_name=timedelta_column_name,
            n_samples=n_samples,
        )
        # to assure mypy that there values are not None to pass to conversion functions:
        assert timedelta_stats["histogram"] is not None
        assert timedelta_stats["max"] is not None
        assert timedelta_stats["mean"] is not None
        assert timedelta_stats["median"] is not None
        assert timedelta_stats["std"] is not None

        datetime_bin_edges = [
            cls.shift_and_convert_to_string(min_date, seconds) for seconds in timedelta_stats["histogram"]["bin_edges"]
        ]

        return DatetimeStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            min=datetime_to_string(min_date),
            max=cls.shift_and_convert_to_string(min_date, timedelta_stats["max"]),
            mean=cls.shift_and_convert_to_string(min_date, timedelta_stats["mean"]),
            median=cls.shift_and_convert_to_string(min_date, timedelta_stats["median"]),
            std=str(datetime.timedelta(seconds=timedelta_stats["std"])),
            histogram=DatetimeHistogram(
                hist=timedelta_stats["histogram"]["hist"],
                bin_edges=datetime_bin_edges,
            ),
        )

    def compute_and_prepare_response(self, data: pl.DataFrame) -> StatisticsPerColumnItem:
        stats = self.compute_statistics(data, column_name=self.name, n_samples=self.n_samples)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.DATETIME,
            column_statistics=stats,
        )


def datetime_to_string(dt: datetime.datetime, format: str = "%Y-%m-%d %H:%M:%S%z") -> str:
    """
    Convert a datetime.datetime object to a string.

    Args:
        dt (datetime): The datetime object to convert.
        format (str, optional): The format of the output string. Defaults to "%Y-%m-%d %H:%M:%S%z".

    Returns:
        str: The datetime object as a string.
    """
    return dt.strftime(format)
