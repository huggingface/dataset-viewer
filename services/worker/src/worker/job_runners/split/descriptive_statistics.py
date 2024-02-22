# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import enum
import logging
import os
from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
import polars as pl
from datasets import Features
from huggingface_hub import hf_hub_download
from libcommon.dtos import JobInfo
from libcommon.exceptions import (
    CacheDirectoryNotInitializedError,
    FeaturesResponseEmptyError,
    NoSupportedFeaturesError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
    StatisticsComputationError,
)
from libcommon.parquet_utils import (
    extract_split_name_from_parquet_url,
    get_num_parquet_files_to_process,
    parquet_export_is_partial,
)
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from requests.exceptions import ReadTimeout

from worker.config import AppConfig, DescriptiveStatisticsConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithCache
from worker.utils import HF_HUB_HTTP_ERROR_RETRY_SLEEPS, retry

REPO_TYPE = "dataset"

DECIMALS = 5
# the maximum number of unique values in a string column so that it is considered to be a category,
# otherwise it's treated as a string
MAX_NUM_STRING_LABELS = 30
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


class Histogram(TypedDict):
    hist: list[int]
    bin_edges: list[Union[int, float]]


class NumericalStatisticsItem(TypedDict):
    nan_count: int
    nan_proportion: float
    min: float
    max: float
    mean: float
    median: float
    std: float
    histogram: Histogram


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


class StatisticsPerColumnItem(TypedDict):
    column_name: str
    column_type: ColumnType
    column_statistics: Union[NumericalStatisticsItem, CategoricalStatisticsItem, BoolStatisticsItem]


class SplitDescriptiveStatisticsResponse(TypedDict):
    num_examples: int
    statistics: list[StatisticsPerColumnItem]
    partial: bool


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
        hist_reverted = hist_df_reverted["reverse_count"].cast(int).to_list()
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


def compute_min_max_median_std_nan_count(df, column_name, n_samples):
    col_stats = dict(
        min=pl.all().min(),
        max=pl.all().max(),
        mean=pl.all().mean(),
        median=pl.all().median(),
        std=pl.all().std(),
        nan_count=pl.all().null_count(),
    )
    stats_names = pl.Series(col_stats.keys())
    stats_expressions = [pl.struct(stat) for stat in col_stats.values()]
    stats = (
        df.select(pl.col(column_name))
        .select(name=stats_names, stats=pl.concat_list(stats_expressions).flatten())
        .unnest("stats")
    )
    minimum, maximum, mean, median, std, nan_count = stats[column_name].to_list()
    minimum, maximum, mean, median, std = np.round([minimum, maximum, mean, median, std], DECIMALS).tolist()
    nan_count = int(nan_count)
    nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0

    return minimum, maximum, mean, median, std, nan_count, nan_proportion


class Column:
    """Abstract class to compute stats for columns of all supported data types. """

    def __init__(self,
                 feature_name: str,
                 data: pl.DataFrame,
                 # feature_info: Optional[dict] = None,
                 ):
        self.name = feature_name
        self.data = data  # ! bevare: data in-memory

    def _compute_statistics(self, *args, **kwargs) -> Union[NumericalStatisticsItem, CategoricalStatisticsItem, BoolStatisticsItem]:
        raise NotImplementedError

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        raise NotImplementedError


class ClassLabelColumn(Column):

    def _compute_statistics(self, n_samples: int, feature_dict: dict, **kwargs) -> CategoricalStatisticsItem:
        datasets_feature = Features.from_dict({self.name: feature_dict})[self.name]
        nan_count = self.data[self.name].null_count()
        nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0

        ids2counts: dict[int, int] = dict(self.data[self.name].value_counts().rows())
        no_label_count = ids2counts.pop(NO_LABEL_VALUE, 0)
        no_label_proportion = np.round(no_label_count / n_samples, DECIMALS).item() if no_label_count != 0 else 0.0

        num_classes = len(datasets_feature.names)
        labels2counts: dict[str, int] = {
            datasets_feature.int2str(cat_id): ids2counts.get(cat_id, 0) for cat_id in range(num_classes)
        }
        n_unique = self.data[self.name].n_unique()
        logging.debug(
            f"{nan_count=} {nan_proportion=} {no_label_count=} {no_label_proportion=} " f"{n_unique=} {labels2counts=}"
        )

        if n_unique > num_classes + int(no_label_count > 0) + int(nan_count > 0):
            raise StatisticsComputationError(
                f"Got unexpected result for ClassLabel {self.name=}: "
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

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        stats = self._compute_statistics(*args, **kwargs)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.CLASS_LABEL,
            column_statistics=stats,
        )


class FloatColumn(Column):

    def _compute_statistics(self, n_bins, n_samples, **kwargs) -> NumericalStatisticsItem:
        minimum, maximum, mean, median, std, nan_count, nan_proportion = compute_min_max_median_std_nan_count(self.data, self.name, n_samples)
        logging.debug(f"{minimum=}, {maximum=}, {mean=}, {median=}, {std=}, {nan_count=} {nan_proportion=}")

        hist = compute_histogram(
            self.data,
            column_name=self.name,
            column_type=ColumnType.FLOAT,
            min_value=minimum,
            max_value=maximum,
            n_bins=n_bins,
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

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        stats = self._compute_statistics(*args, **kwargs)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.FLOAT,
            column_statistics=stats,
        )


class IntColumn(Column):

    def _compute_statistics(self, n_bins: int, n_samples: int, **kwargs) -> NumericalStatisticsItem:
        logging.info(f"Compute statistics for integer column {self.name} with polars. ")
        minimum, maximum, mean, median, std, nan_count, nan_proportion = compute_min_max_median_std_nan_count(self.data, self.name, n_samples)
        minimum, maximum = int(minimum), int(maximum)

        hist = compute_histogram(
            self.data,
            column_name=self.name,
            column_type=ColumnType.INT,
            min_value=minimum,
            max_value=maximum,
            n_bins=n_bins,
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

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        stats = self._compute_statistics(*args, **kwargs)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.INT,
            column_statistics=stats,
        )


class StringColumn(Column):

    def _compute_statistics(self, n_bins: int, n_samples: int, **kwargs) -> Union[CategoricalStatisticsItem, NumericalStatisticsItem]:
        # TODO: count n_unique only on the first parquet file?
        n_unique = self.data[self.name].n_unique()
        nan_count = self.data[self.name].null_count()
        nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0

        if n_unique <= MAX_NUM_STRING_LABELS:  # TODO: make relative
            labels2counts: dict[str, int] = dict(self.data[self.name].value_counts().rows())
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

        lengths_column_name = f"{self.name}_len"
        lengths_df = self.data.select(pl.col(self.name)).with_columns(
            pl.col(self.name).str.len_chars().alias(lengths_column_name)
        )
        lengths_column = IntColumn(lengths_column_name, lengths_df)
        return lengths_column._compute_statistics(
            n_bins=n_bins,
            n_samples=n_samples,
        )

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        stats = self._compute_statistics(*args, **kwargs)
        type = ColumnType.STRING_LABEL if "frequencies" in stats else ColumnType.STRING_TEXT
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=type,
            column_statistics=stats,
        )


class BoolColumn(Column):

    def _compute_statistics(self, n_samples: int, **kwargs) -> BoolStatisticsItem:
        # df = pl.read_parquet(self.path, columns=[self.name])
        nan_count = self.data[self.name].null_count()
        nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0
        values2counts: dict[str, int] = dict(self.data[self.name].value_counts().rows())
        # exclude counts of None values from frequencies if exist:
        values2counts.pop(None, None)  # type: ignore

        return BoolStatisticsItem(
            nan_count=nan_count,
            nan_proportion=nan_proportion,
            frequencies={str(key): freq for key, freq in sorted(values2counts.items())},
        )

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        stats = self._compute_statistics(*args, **kwargs)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.BOOL,
            column_statistics=stats,
        )


class ListColumn(Column):

    def _compute_statistics(self, n_bins: int, n_samples: int, **kwargs) -> NumericalStatisticsItem:
        nan_count = self.data[self.name].null_count()
        nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0
        df_without_na = self.data.select(pl.col(self.name)).drop_nulls()

        lengths_column_name = f"{self.name}_len"
        lengths_df = df_without_na.with_columns(pl.col(self.name).list.len().alias(lengths_column_name))
        lengths_column = IntColumn(lengths_column_name, lengths_df)
        lengths_stats = lengths_column._compute_statistics(n_bins=n_bins, n_samples=n_samples - nan_count)
        lengths_stats["nan_count"] = nan_count
        lengths_stats["nan_proportion"] = nan_proportion

        return NumericalStatisticsItem(
            **lengths_stats,
        )

    def prepare_response(self, *args, **kwargs) -> StatisticsPerColumnItem:
        stats = self._compute_statistics(*args, **kwargs)
        return StatisticsPerColumnItem(
            column_name=self.name,
            column_type=ColumnType.LIST,
            column_statistics=stats,
        )


SupportedColumns = type[Union[IntColumn, FloatColumn, ClassLabelColumn, StringColumn, BoolColumn, ListColumn]]


def compute_descriptive_statistics_response(
    dataset: str,
    config: str,
    split: str,
    local_parquet_directory: Path,
    hf_token: Optional[str],
    parquet_revision: str,
    histogram_num_bins: int,
    max_split_size_bytes: int,
    parquet_metadata_directory: StrPath,
) -> SplitDescriptiveStatisticsResponse:
    """
    Get the response of 'split-descriptive-statistics' for one specific split of a dataset from huggingface.co.
    Currently, integers, floats and ClassLabel features are supported.

    Args:
        dataset (`str`):
            Name of a dataset.
        config (`str`):
            Requested dataset configuration name.
        split (`str`):
            Requested dataset split.
        local_parquet_directory (`Path`):
            Path to a local directory where the dataset's parquet files are stored. We download these files locally
            because it enables fast querying and statistics computation.
        hf_token (`str`, *optional*):
            An app authentication token with read access to all the datasets.
        parquet_revision (`str`):
            The git revision (e.g. "refs/convert/parquet") from where to download the dataset's parquet files.
        histogram_num_bins (`int`):
            (Maximum) number of bins to compute histogram for numerical data.
            The resulting number of bins might be lower than the requested one for integer data.
        max_split_size_bytes (`int`):
            If raw uncompressed split data is larger than this value, the statistics are computed
            only on the first parquet files, approximately up to this size, and the `partial` field will be set
            to `True` in the response.
        parquet_metadata_directory (`StrPath`):
            Path to directory on local shared storage containing parquet metadata files. Parquet metadata is needed
            to get uncompressed size of split files to determine the number of files to use if split is larger
            than `max_split_size_bytes`

    Raises:
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step does not have the expected format.
        [~`libcommon.exceptions.ParquetResponseEmptyError`]:
            If response for `config-parquet-and-info` doesn't have any parquet files.
        [~`libcommon.exceptions.FeaturesResponseEmptyError`]:
            If response for `config-parquet-and-info` doesn't have features.
        [~`libcommon.exceptions.NoSupportedFeaturesError`]:
            If requested dataset doesn't have any supported for statistics computation features.
            Currently, floats, integers and ClassLabels are supported.
        [~`libcommon.exceptions.StatisticsComputationError`]:
            If there was some unexpected behaviour during statistics computation.

    Returns:
        `SplitDescriptiveStatisticsResponse`: An object with the statistics response for a requested split, per each
            numerical (int and float) or ClassLabel feature.
    """

    logging.info(f"compute 'split-descriptive-statistics' for {dataset=} {config=} {split=}")

    # get parquet urls and dataset_info
    config_parquet_metadata_step = "config-parquet-metadata"
    parquet_metadata_best_response = get_previous_step_or_raise(
        kinds=[config_parquet_metadata_step],
        dataset=dataset,
        config=config,
    )
    content_parquet_metadata = parquet_metadata_best_response.response["content"]
    try:
        split_parquet_files = [
            parquet_file
            for parquet_file in content_parquet_metadata["parquet_files_metadata"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]
        features = content_parquet_metadata["features"]

    except KeyError as e:
        raise PreviousStepFormatError(
            f"Previous step '{config_parquet_metadata_step}' did not return the expected content", e
        ) from e

    if not split_parquet_files:
        raise ParquetResponseEmptyError("No parquet files found.")

    if not features:
        raise FeaturesResponseEmptyError("No features found.")

    num_parquet_files_to_process, num_bytes, num_rows = get_num_parquet_files_to_process(
        parquet_files=split_parquet_files,
        parquet_metadata_directory=parquet_metadata_directory,
        max_size_bytes=max_split_size_bytes,
    )
    partial_parquet_export = parquet_export_is_partial(split_parquet_files[0]["url"])
    partial = partial_parquet_export or (num_parquet_files_to_process < len(split_parquet_files))
    split_parquet_files = split_parquet_files[:num_parquet_files_to_process]

    # store data as local parquet files for fast querying
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    logging.info(f"Downloading remote parquet files to a local directory {local_parquet_directory}. ")
    # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
    # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
    split_directory = extract_split_name_from_parquet_url(split_parquet_files[0]["url"])
    for parquet_file in split_parquet_files:
        retry_download_hub_file = retry(on=[ReadTimeout], sleeps=HF_HUB_HTTP_ERROR_RETRY_SLEEPS)(hf_hub_download)
        retry_download_hub_file(
            repo_type=REPO_TYPE,
            revision=parquet_revision,
            repo_id=dataset,
            filename=f"{config}/{split_directory}/{parquet_file['filename']}",
            local_dir=local_parquet_directory,
            local_dir_use_symlinks=False,
            token=hf_token,
            cache_dir=local_parquet_directory,
            force_download=True,
            resume_download=False,
        )

    local_parquet_glob_path = Path(local_parquet_directory) / config / f"{split_directory}/*.parquet"

    num_examples = pl.read_parquet(
        local_parquet_glob_path, columns=[pl.scan_parquet(local_parquet_glob_path).columns[0]]
    ).shape[0]

    def _feature_type(dataset_feature) -> Optional[SupportedColumns]:
        if isinstance(dataset_feature, dict):
            if dataset_feature.get("_type") == "ClassLabel":
                return ClassLabelColumn
            if dataset_feature.get("_type") == "Value" and dataset_feature.get("dtype") in INTEGER_DTYPES:
                return IntColumn
            if dataset_feature.get("_type") == "Value" and dataset_feature.get("dtype") in FLOAT_DTYPES:
                return FloatColumn
            if dataset_feature.get("_type") == "Value" and dataset_feature.get("dtype") in STRING_DTYPES:
                return StringColumn
            if dataset_feature.get("_type") == "Value" and dataset_feature.get("dtype") == "bool":
                return BoolColumn
            if dataset_feature.get("_type") == "Sequence":
                return ListColumn
        if isinstance(dataset_feature, list):
            return ListColumn

    all_stats = []
    for feature_name, feature in features.items():
        stats_feature_type = _feature_type(feature)
        if stats_feature_type:
            data = pl.read_parquet(local_parquet_glob_path, columns=[feature_name])
            stats_feature = stats_feature_type(feature_name, data)
            column_stats = stats_feature.prepare_response(
                n_bins=histogram_num_bins,
                n_samples=num_examples,
                feature_dict=feature,
            )
            all_stats.append(column_stats)
    if not all_stats:
        raise NoSupportedFeaturesError(
            "No columns for statistics computation found. Currently supported feature types are: "
            f"{NUMERICAL_DTYPES}, {STRING_DTYPES}, ClassLabel, list/Sequence and bool. "
        )

    return SplitDescriptiveStatisticsResponse(
        num_examples=num_examples,
        statistics=sorted(all_stats, key=lambda x: x["column_name"]),
        partial=partial,
    )


class SplitDescriptiveStatisticsJobRunner(SplitJobRunnerWithCache):
    descriptive_statistics_config: DescriptiveStatisticsConfig

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        statistics_cache_directory: StrPath,
        parquet_metadata_directory: StrPath,
    ):
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            cache_directory=Path(statistics_cache_directory),
        )
        self.descriptive_statistics_config = app_config.descriptive_statistics
        self.parquet_metadata_directory = parquet_metadata_directory

    @staticmethod
    def get_job_type() -> str:
        return "split-descriptive-statistics"

    def compute(self) -> CompleteJobResult:
        if self.cache_subdirectory is None:
            raise CacheDirectoryNotInitializedError("Cache directory has not been initialized.")
        return CompleteJobResult(
            compute_descriptive_statistics_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                local_parquet_directory=self.cache_subdirectory,
                hf_token=self.app_config.common.hf_token,
                parquet_revision=self.descriptive_statistics_config.parquet_revision,
                histogram_num_bins=self.descriptive_statistics_config.histogram_num_bins,
                max_split_size_bytes=self.descriptive_statistics_config.max_split_size_bytes,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        )
