# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import enum
import logging
import os
from pathlib import Path
from typing import Optional, TypedDict, Union

import duckdb
import numpy as np
import pandas as pd
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

DATA_TABLE_NAME = "data"
BINS_TABLE_NAME = "bins"  # name of a table with bin edges data used to compute histogram
STRING_LENGTHS_TABLE_NAME = "string_lengths"


COMPUTE_NAN_COUNTS_COMMAND = """
    SELECT COUNT(*) FROM {data_table_name} WHERE "{column_name}" IS NULL;
"""
COMPUTE_CATEGORIES_COUNTS_COMMAND = """
    SELECT "{column_name}", COUNT(*) FROM {data_table_name} GROUP BY "{column_name}";
"""
COMPUTE_MIN_MAX_MEAN_MEDIAN_STD_COMMAND = """
    SELECT min("{column_name}"), max("{column_name}"), mean("{column_name}"),
    median("{column_name}"), stddev_samp("{column_name}") FROM {data_table_name};
"""
COMPUTE_HIST_COMMAND = """
    SELECT bin_id, COUNT(*) as count FROM {data_table_name}
        JOIN {bins_table_name} ON ("{column_name}" >= bin_min AND "{column_name}" < bin_max) GROUP BY bin_id;
"""
CREATE_TABLE_COMMAND = """
CREATE OR REPLACE TABLE {table_name} AS SELECT {column_names} FROM {select_from};
"""
CREATE_TEMPORARY_TABLE_COMMAND = """
CREATE OR REPLACE TEMPORARY TABLE {table_name} AS SELECT {column_names} FROM {select_from};
"""


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
) -> pd.DataFrame:
    """
    Returns:
        pandas.DataFrame with bin edges to insert into database to perform histogram computation with duckdb
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
    bin_max_edges = bin_edges[1:] + [max_value + 1]  # add 1 to include exact max values in the last bin
    return pd.DataFrame.from_dict(
        {"bin_id": list(range(len(bin_edges))), "bin_min": bin_edges, "bin_max": bin_max_edges}
    )


def compute_histogram(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    table_name: str,
    column_type: ColumnType,
    min_value: Union[int, float],
    max_value: Union[int, float],
    n_bins: int,
    n_samples: Optional[int] = None,
) -> Histogram:
    bins_df = generate_bins(
        min_value=min_value, max_value=max_value, column_name=column_name, column_type=column_type, n_bins=n_bins
    )
    n_bins = bins_df.shape[0]
    # create auxiliary table with bin edges
    con.sql(CREATE_TEMPORARY_TABLE_COMMAND.format(table_name=BINS_TABLE_NAME, column_names="*", select_from="bins_df"))
    compute_hist_command = COMPUTE_HIST_COMMAND.format(
        data_table_name=table_name, bins_table_name=BINS_TABLE_NAME, column_name=column_name
    )
    logging.debug(f"Compute histogram for {column_name=}.\n{compute_hist_command}")
    # query returns list of tuples (bin_id, bin_max, n_count):
    hist_query_result = dict(con.sql(compute_hist_command).fetchall())  # dict bin_id -> n_samples
    if len(hist_query_result) > n_bins + 1:
        raise StatisticsComputationError(
            f"Got unexpected result during histogram computation for {column_name=}: returned more bins than"
            f" requested. {n_bins=} {hist_query_result=}. "
        )
    hist = []
    for bin_idx in range(n_bins):
        # no key in query result = no examples in this range, so we put 0
        hist.append(hist_query_result.get(bin_idx, 0))
    if n_samples and sum(hist) != n_samples:
        raise StatisticsComputationError(
            f"Got unexpected result during histogram computation for {column_name=}: "
            f" histogram sum and number of non-null samples don't match, histogram sum={sum(hist)}, {n_samples=}"
        )
    bins = bins_df["bin_min"].round(DECIMALS).tolist()
    bins = bins + [np.round(max_value, DECIMALS).item()]  # put exact max value back to bins
    return Histogram(hist=hist, bin_edges=bins)


def compute_numerical_statistics(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    table_name: str,
    n_bins: int,
    n_samples: int,
    column_type: ColumnType,
) -> NumericalStatisticsItem:
    min_max_mean_median_std_command = COMPUTE_MIN_MAX_MEAN_MEDIAN_STD_COMMAND.format(
        column_name=column_name, data_table_name=table_name
    )
    logging.debug(
        f"Compute min, max, mean, median, std and proportion of null values for {column_name=}. "
        f"{min_max_mean_median_std_command}"
    )
    minimum, maximum, mean, median, std = con.sql(min_max_mean_median_std_command).fetchall()[0]
    logging.debug(f"{minimum=}, {maximum=}, {mean=}, {median=}, {std=}")

    nan_count_command = COMPUTE_NAN_COUNTS_COMMAND.format(column_name=column_name, data_table_name=table_name)
    nan_count = con.sql(nan_count_command).fetchall()[0][0]
    nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0
    logging.debug(f"{nan_count=} {nan_proportion=}")
    histogram = compute_histogram(
        con,
        column_name,
        table_name,
        min_value=minimum,
        max_value=maximum,
        column_type=column_type,
        n_bins=n_bins,
        n_samples=n_samples - nan_count,
    )
    if column_type == ColumnType.FLOAT:
        minimum, maximum, mean, median, std = np.round([minimum, maximum, mean, median, std], DECIMALS).tolist()
    elif column_type == ColumnType.INT:
        mean, median, std = np.round([mean, median, std], DECIMALS).tolist()
    else:
        raise ValueError(f"Incorrect column type of {column_name=}: {column_type}")
    return NumericalStatisticsItem(
        nan_count=nan_count,
        nan_proportion=nan_proportion,
        min=minimum,
        max=maximum,
        mean=mean,
        median=median,
        std=std,
        histogram=histogram,
    )


def compute_categorical_statistics(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    table_name: str,
    class_label_feature: ClassLabel,
    n_samples: int,
) -> CategoricalStatisticsItem:
    categorical_counts_query = COMPUTE_CATEGORIES_COUNTS_COMMAND.format(
        column_name=column_name, data_table_name=table_name
    )
    logging.debug(f"Compute categories counts for {column_name}.\n{categorical_counts_query}")
    ids2counts: dict[Optional[int], int] = dict(
        con.sql(categorical_counts_query).fetchall()
    )  # dict {idx: num_samples}; idx might be also None for null values and -1 for `no label`
    nan_count = ids2counts.pop(None, 0)
    no_label_count = ids2counts.pop(NO_LABEL_VALUE, 0)
    num_classes = len(class_label_feature.names)
    labels2counts: dict[str, int] = {
        class_label_feature.int2str(cat_id): ids2counts.get(cat_id, 0) for cat_id in range(num_classes)
    }
    n_unique_computed = len(ids2counts)
    if n_unique_computed > num_classes:
        raise StatisticsComputationError(
            f"Got unexpected result during classes distribution computation for {column_name=}: computed number of"
            f" classes don't match with feature's num_classes. {n_unique_computed=} {num_classes=}. "
        )
    no_label_proportion = np.round(no_label_count / n_samples, DECIMALS).item() if no_label_count != 0 else 0.0
    nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0
    logging.debug(
        f"{nan_count=}, {nan_proportion=}, {n_unique_computed}, {no_label_count=}, {no_label_proportion=},"
        f" frequencies={labels2counts}."
    )

    return CategoricalStatisticsItem(
        nan_count=nan_count,
        nan_proportion=nan_proportion,
        no_label_count=no_label_count,
        no_label_proportion=no_label_proportion,
        n_unique=num_classes,
        frequencies=labels2counts,
    )


def compute_string_statistics(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    n_bins: int,
    n_samples: int,
    table_name: str,
    dtype: Optional[str],
) -> Union[CategoricalStatisticsItem, NumericalStatisticsItem]:
    if dtype != "large_string":
        categorical_counts_query = COMPUTE_CATEGORIES_COUNTS_COMMAND.format(
            column_name=column_name, data_table_name=table_name
        )
        logging.debug(f"Compute categories counts for {column_name=}.\n{categorical_counts_query}")
        labels2counts: dict[str, int] = dict(con.sql(categorical_counts_query).fetchall())
        n_unique = len(labels2counts)
        if n_unique <= MAX_NUM_STRING_LABELS:
            # consider string as categories
            nan_count = labels2counts.pop(None, 0)  # type: ignore
            nan_proportion = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0
            logging.debug(
                "Treat column as category. "
                f"{nan_count=}, {nan_proportion=}, {n_unique=}, frequencies={labels2counts}. "
            )
            return CategoricalStatisticsItem(
                nan_count=nan_count,
                nan_proportion=nan_proportion,
                no_label_count=0,
                no_label_proportion=0.0,
                n_unique=len(labels2counts),
                frequencies=labels2counts,
            )
    # compute numerical stats over string lengths (min, max, ..., hist)
    string_lengths_column_name = f"{column_name}__lengths"
    logging.debug(f"Treat {column_name=} as string and compute numerical stats over its lengths.")
    con.sql(
        CREATE_TEMPORARY_TABLE_COMMAND.format(
            table_name=STRING_LENGTHS_TABLE_NAME,
            column_names=f'length("{column_name}") AS "{string_lengths_column_name}"',
            select_from=table_name,
        )
    )
    return compute_numerical_statistics(
        con=con,
        column_name=string_lengths_column_name,
        table_name=STRING_LENGTHS_TABLE_NAME,
        n_bins=n_bins,
        n_samples=n_samples,
        column_type=ColumnType.INT,
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
    for parquet_file in split_parquet_files:
        # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
        # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
        split_directory = parquet_file["url"].rsplit("/", 2)[1]
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

    local_parquet_glob_path = Path(local_parquet_directory) / config / f"{split}/*.parquet"

    stats: list[StatisticsPerColumnItem] = []
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
    all_feature_names = ",".join(
        f'"{column}"' for column in list(categorical_features) + list(numerical_features) + list(string_features)
    )

    con = duckdb.connect(":memory:")  # we don't load data in local db file, we load it in an in-memory table
    con.sql("SET enable_progress_bar=true;")
    n_threads = con.sql("SELECT current_setting('threads')").fetchall()[0][0]
    logging.info(f"Original number of threads={n_threads}")
    con.sql("SET threads TO 8;")
    n_threads = con.sql("SELECT current_setting('threads')").fetchall()[0][0]
    logging.info(f"Number of threads={n_threads}")

    logging.info("Loading data into in-memory table. ")
    create_table_command = CREATE_TABLE_COMMAND.format(
        table_name=DATA_TABLE_NAME,
        column_names=all_feature_names,
        select_from=f"read_parquet('{local_parquet_glob_path}')",
    )
    logging.info(create_table_command)
    con.sql(create_table_command)
    logging.info("Loading finished. ")

    if string_features:
        logging.info(f"Compute statistics for string columns {string_features}")
    for feature_name, feature in tqdm(string_features.items()):
        logging.debug(f"Compute for string column {feature_name}")
        string_column_stats = compute_string_statistics(
            con,
            feature_name,
            n_bins=histogram_num_bins,
            n_samples=num_examples,
            table_name=DATA_TABLE_NAME,
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
    # compute for ClassLabels (we are sure that these are discrete categories)
    if categorical_features:
        logging.info(f"Compute statistics for categorical columns {categorical_features}")
        categorical_features = Features.from_dict(categorical_features)
    for feature_name, feature in tqdm(categorical_features.items()):
        logging.debug(f"Compute statistics for ClassLabel feature '{feature_name}'")
        cat_column_stats: CategoricalStatisticsItem = compute_categorical_statistics(
            con,
            feature_name,
            class_label_feature=feature,
            n_samples=num_examples,
            table_name=DATA_TABLE_NAME,
        )
        stats.append(
            StatisticsPerColumnItem(
                column_name=feature_name,
                column_type=ColumnType.CLASS_LABEL,
                column_statistics=cat_column_stats,
            )
        )

    if numerical_features:
        logging.info(f"Compute min, max, mean, median, std, histogram for numerical columns {numerical_features}. ")
    for feature_name, feature in tqdm(numerical_features.items()):
        column_type = ColumnType.FLOAT if feature["dtype"] in FLOAT_DTYPES else ColumnType.INT
        num_column_stats: NumericalStatisticsItem = compute_numerical_statistics(
            con,
            feature_name,
            table_name=DATA_TABLE_NAME,
            n_bins=histogram_num_bins,
            n_samples=num_examples,
            column_type=column_type,
        )
        stats.append(
            StatisticsPerColumnItem(
                column_name=feature_name,
                column_type=column_type,
                column_statistics=num_column_stats,
            )
        )
    con.close()

    return SplitDescriptiveStatisticsResponse(
        num_examples=num_examples, statistics=sorted(stats, key=lambda x: x["column_name"])
    )


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
