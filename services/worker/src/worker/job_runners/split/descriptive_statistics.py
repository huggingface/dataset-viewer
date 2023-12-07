# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import enum
import logging
import os
from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
import polars as pl
from datasets import ClassLabel, Features
from huggingface_hub import hf_hub_download
from libcommon.constants import PROCESSING_STEP_SPLIT_DESCRIPTIVE_STATISTICS_VERSION
from libcommon.exceptions import (
    CacheDirectoryNotInitializedError,
    NoSupportedFeaturesError,
    ParquetResponseEmptyError,
    PreviousStepFormatError,
    SplitWithTooBigParquetError,
    StatisticsComputationError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.utils import JobInfo
from tqdm import tqdm

from worker.config import AppConfig, DescriptiveStatisticsConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithCache

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
    CLASS_LABEL = "class_label"
    STRING_LABEL = "string_label"
    STRING_TEXT = "string_text"


class Histogram(TypedDict):
    hist: list[int]
    bin_edges: list[float]


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


class StatisticsPerColumnItem(TypedDict):
    column_name: str
    column_type: ColumnType
    column_statistics: Union[NumericalStatisticsItem, CategoricalStatisticsItem]


class SplitDescriptiveStatisticsResponse(TypedDict):
    num_examples: int
    statistics: list[StatisticsPerColumnItem]


def generate_bins(
    min_value: Union[int, float],
    max_value: Union[int, float],
    column_name: str,
    column_type: ColumnType,
    n_bins: int,
) -> list[int]:
    """
    Returns:
        List of bin edges.
    """
    if column_type is ColumnType.FLOAT:
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


def compute_histogram_polars(
    df: pl.dataframe.frame.DataFrame,
    column_name: str,
    column_type: ColumnType,
    min_value: Union[int, float],
    max_value: Union[int, float],
    n_bins: int,
    n_samples: int,
) -> Histogram:
    bins = generate_bins(
        min_value=min_value, max_value=max_value, column_name=column_name, column_type=column_type, n_bins=n_bins
    )
    bins_reverted = [-1 * b for b in bins[::-1]]
    hist_df_reverted = df.with_columns(pl.col(column_name).mul(-1).alias("reverse"))["reverse"].hist(
        bins=bins_reverted
    )
    hist_reverted = hist_df_reverted["reverse_count"].cast(int).to_list()  # TODO: assert the last one is 0
    # assert hist_reverted[-1] == 0  # this value corresponds to "less than minimum"
    hist = hist_reverted[::-1]
    hist = [hist[0] + hist[1]] + hist[2:-2] + [hist[-2] + hist[-1]]
    if n_samples and sum(hist) != n_samples:
        raise StatisticsComputationError(
            f"Got unexpected result during histogram computation for {column_name=}: "
            f" histogram sum and number of non-null samples don't match, histogram sum={sum(hist)}, {n_samples=}"
        )
    return Histogram(hist=hist, bin_edges=np.round(bins, DECIMALS).tolist())


def compute_numerical_statistics_polars(
    df: pl.dataframe.frame.DataFrame,
    column_name: str,
    n_bins: int,
    n_samples: int,  # TODO: partial datasets
    column_type: ColumnType,
) -> NumericalStatisticsItem:
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
    nan_count = int(nan_count)

    if column_type == ColumnType.FLOAT:
        minimum, maximum, mean, median, std = np.round([minimum, maximum, mean, median, std], DECIMALS).tolist()
    elif column_type == ColumnType.INT:
        mean, median, std = np.round([mean, median, std], DECIMALS).tolist()
        minimum, maximum = int(minimum), int(maximum)
    else:
        raise ValueError(f"Incorrect column type of {column_name=}: {column_type}")
    nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0

    hist = compute_histogram_polars(
        df,
        column_name=column_name,
        column_type=column_type,
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


def compute_categorical_statistics_polars(
    df: pl.dataframe.frame.DataFrame,
    column_name: str,
    class_label_feature: ClassLabel,
    n_samples: int,
) -> CategoricalStatisticsItem:
    nan_count = df[column_name].null_count()
    nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0

    ids2counts: dict[int, int] = dict(df[column_name].value_counts().rows())  # type: ignore
    no_label_count = ids2counts.pop(NO_LABEL_VALUE, 0)
    no_label_proportion = np.round(no_label_count / n_samples, DECIMALS).item() if no_label_count != 0 else 0.0

    num_classes = len(class_label_feature.names)
    labels2counts: dict[str, int] = {
        class_label_feature.int2str(cat_id): ids2counts.get(cat_id, 0) for cat_id in range(num_classes)
    }
    # n_unique = df[column_name].n_unique()
    # if not n_unique == num_classes + int(no_label_count > 0):
    #     raise

    return CategoricalStatisticsItem(
        nan_count=nan_count,
        nan_proportion=nan_proportion,
        no_label_count=no_label_count,
        no_label_proportion=no_label_proportion,
        n_unique=num_classes,  # TODO: does it contain None?
        frequencies=labels2counts,
    )


def compute_string_statistics_polars(
    df: pl.dataframe.frame.DataFrame,
    column_name: str,
    n_bins: int,
    n_samples: int,
    dtype: Optional[str],
) -> Union[CategoricalStatisticsItem, NumericalStatisticsItem]:
    if dtype != "large_string":
        n_unique = df[column_name].n_unique()
        nan_count = df[column_name].null_count()
        nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0

        if n_unique <= MAX_NUM_STRING_LABELS:
            labels2counts: dict[str, int] = dict(df[column_name].value_counts().rows())  # type: ignore
            return CategoricalStatisticsItem(
                nan_count=nan_count,
                nan_proportion=nan_proportion,
                no_label_count=0,
                no_label_proportion=0.0,
                n_unique=len(labels2counts),  # TODO: does it contain None?
                frequencies=labels2counts,
            )

    lengths_df = df.select(pl.col(column_name)).with_columns(
        pl.col(column_name).str.len_chars().alias(f"{column_name}_len")
    )
    return compute_numerical_statistics_polars(
        df=lengths_df,
        column_name=f"{column_name}_len",
        n_bins=n_bins,
        n_samples=n_samples,
        column_type=ColumnType.INT,
    )


def compute_descriptive_statistics_response_polars(
    path: Path,
    string_features: Optional[dict[str, dict]],  # type: ignore
    categorical_features: Optional[dict[str, dict]],  # type: ignore
    numerical_features: Optional[dict[str, dict]],  # type: ignore
    histogram_num_bins: int,
    num_examples: int,
) -> SplitDescriptiveStatisticsResponse:
    stats = []

    if string_features:
        logging.info(f"Compute statistics for string columns {string_features} with POLARS")
        for feature_name, feature in tqdm(string_features.items()):
            logging.info(f"Compute for string column '{feature_name}'")
            df = pl.read_parquet(path, columns=[feature_name])
            string_column_stats = compute_string_statistics_polars(
                df,
                feature_name,
                n_bins=histogram_num_bins,
                n_samples=num_examples,
                dtype=feature.get("dtype"),
            )
            stats.append(
                StatisticsPerColumnItem(
                    column_name=feature_name,
                    column_type=ColumnType.STRING_LABEL
                    if "frequencies" in string_column_stats
                    else ColumnType.STRING_TEXT,
                    column_statistics=string_column_stats,
                )
            )

    # # compute for ClassLabels (we are sure that these are discrete categories)
    if categorical_features:
        logging.info(f"Compute statistics for categorical columns {categorical_features}  with POLARS")
        categorical_features = Features.from_dict(categorical_features)
        for feature_name, feature in tqdm(categorical_features.items()):  # type: ignore
            logging.debug(f"Compute statistics for ClassLabel feature '{feature_name}'")
            df = pl.read_parquet(path, columns=[feature_name])
            cat_column_stats: CategoricalStatisticsItem = compute_categorical_statistics_polars(
                df,
                feature_name,
                class_label_feature=feature,
                n_samples=num_examples,
            )
            stats.append(
                StatisticsPerColumnItem(
                    column_name=feature_name,
                    column_type=ColumnType.CLASS_LABEL,
                    column_statistics=cat_column_stats,
                )
            )

    if numerical_features:
        logging.info(f"Compute statistics for numerical columns {numerical_features} with POLARS")
        for feature_name, feature in tqdm(numerical_features.items()):
            logging.info(f"Compute for numerical column '{feature_name}'")
            column_type = ColumnType.FLOAT if feature["dtype"] in FLOAT_DTYPES else ColumnType.INT
            df = pl.read_parquet(path, columns=[feature_name])
            numerical_column_stats = compute_numerical_statistics_polars(
                df,
                column_name=feature_name,
                n_bins=histogram_num_bins,
                n_samples=num_examples,
                column_type=column_type,
            )
            stats.append(
                StatisticsPerColumnItem(
                    column_name=feature_name,
                    column_type=column_type,
                    column_statistics=numerical_column_stats,
                )
            )
    return SplitDescriptiveStatisticsResponse(
        num_examples=num_examples, statistics=sorted(stats, key=lambda x: x["column_name"])
    )


def compute_descriptive_statistics_response(
    dataset: str,
    config: str,
    split: str,
    local_parquet_directory: Path,
    hf_token: Optional[str],
    parquet_revision: str,
    histogram_num_bins: int,
    max_parquet_size_bytes: int,
) -> SplitDescriptiveStatisticsResponse:
    """
    Compute statistics and get response for the `split-descriptive-statistics` step.
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
        hf_token (`str`, `optional`):
            An app authentication token with read access to all the datasets.
        parquet_revision (`str`):
            The git revision (e.g. "refs/convert/parquet") from where to download the dataset's parquet files.
        histogram_num_bins (`int`):
            (Maximum) number of bins to compute histogram for numerical data.
            The resulting number of bins might be lower than the requested one for integer data.
        max_parquet_size_bytes (`int`):
            The maximum size in bytes of the dataset's parquet files to compute statistics.
            Datasets with bigger size are ignored.

    Returns:
        `SplitDescriptiveStatisticsResponse`: An object with the statistics response for a requested split, per each
        numerical (int and float) or ClassLabel feature.

    Raises the following errors:
        - [`libcommon.exceptions.PreviousStepFormatError`]
            If the content of the previous step does not have the expected format.
        - [`libcommon.exceptions.ParquetResponseEmptyError`]
            If response for `config-parquet-and-info` doesn't have any parquet files.
        - [`libcommon.exceptions.SplitWithTooBigParquetError`]
            If requested split's parquet files size exceeds the provided `max_parquet_size_bytes`.
        - [`libcommon.exceptions.NoSupportedFeaturesError`]
            If requested dataset doesn't have any supported for statistics computation features.
            Currently, floats, integers and ClassLabels are supported.
        - [`libcommon.exceptions.StatisticsComputationError`]
            If there was some unexpected behaviour during statistics computation.
    """

    logging.info(f"Compute descriptive statistics for {dataset=}, {config=}, {split=}")

    config_parquet_and_info_step = "config-parquet-and-info"
    parquet_and_info_best_response = get_previous_step_or_raise(
        kinds=[config_parquet_and_info_step],
        dataset=dataset,
        config=config,
    )
    content_parquet_and_info = parquet_and_info_best_response.response["content"]
    try:
        split_parquet_files = [
            parquet_file
            for parquet_file in content_parquet_and_info["parquet_files"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]
        dataset_info = content_parquet_and_info["dataset_info"]
    except KeyError as e:
        raise PreviousStepFormatError(
            (
                f"Previous step '{config_parquet_and_info_step}' did not return the expected content: "
                "'parquet_files' or 'dataset_info'. "
            ),
            e,
        ) from e

    if not split_parquet_files:
        raise ParquetResponseEmptyError("No parquet files found.")
    features = dataset_info.get("features")
    if features is None:
        raise PreviousStepFormatError(
            f"Previous step '{config_parquet_and_info_step}' did not return the expected content: "
            "no features found in 'dataset_info'. "
        )

    split_parquets_size = sum(parquet_file["size"] for parquet_file in split_parquet_files)
    if split_parquets_size > max_parquet_size_bytes:
        raise SplitWithTooBigParquetError(
            f"Statistics computation is limited to split parquets under {max_parquet_size_bytes} bytes. "
            f"Current size of sum of split parquets is {split_parquets_size} bytes."
        )

    # store data as local parquet files for fast querying
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    logging.info(f"Downloading remote parquet files to a local directory {local_parquet_directory}. ")
    # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
    # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
    split_directory = split_parquet_files[0]["url"].rsplit("/", 2)[1]
    for parquet_file in split_parquet_files:
        hf_hub_download(
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

    num_examples = dataset_info["splits"][split]["num_examples"]
    categorical_features = {
        feature_name: feature
        for feature_name, feature in features.items()
        if isinstance(feature, dict) and feature.get("_type") == "ClassLabel"
    }
    numerical_features = {
        feature_name: feature
        for feature_name, feature in features.items()
        if isinstance(feature, dict) and feature.get("_type") == "Value" and feature.get("dtype") in NUMERICAL_DTYPES
    }
    string_features = {
        feature_name: feature
        for feature_name, feature in features.items()
        if isinstance(feature, dict) and feature.get("_type") == "Value" and feature.get("dtype") in STRING_DTYPES
    }
    if not categorical_features and not numerical_features and not string_features:
        raise NoSupportedFeaturesError(
            "No columns for statistics computation found. Currently supported feature types are: "
            f"{NUMERICAL_DTYPES}, {STRING_DTYPES} and ClassLabel. "
        )

    logging.info(f"Compute statistics for {dataset=} {config=} {split=} with POLARS")
    response = compute_descriptive_statistics_response_polars(
        path=local_parquet_glob_path,
        string_features=string_features,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        num_examples=num_examples,
        histogram_num_bins=histogram_num_bins,
    )
    return response


class SplitDescriptiveStatisticsJobRunner(SplitJobRunnerWithCache):
    descriptive_statistics_config: DescriptiveStatisticsConfig

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        statistics_cache_directory: StrPath,
    ):
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            cache_directory=Path(statistics_cache_directory),
        )
        self.descriptive_statistics_config = app_config.descriptive_statistics

    @staticmethod
    def get_job_type() -> str:
        return "split-descriptive-statistics"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_DESCRIPTIVE_STATISTICS_VERSION

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
                max_parquet_size_bytes=self.descriptive_statistics_config.max_parquet_size_bytes,
            )
        )
