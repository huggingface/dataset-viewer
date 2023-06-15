# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import duckdb
import numpy as np
from libcommon.constants import PROCESSING_STEP_SPLIT_DESCRIPTIVE_STATS_VERSION
from libcommon.exceptions import PreviousStepFormatError

# from libcommon.exceptions import SplitWithTooBigParquetError
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import JobInfo
from tqdm import tqdm

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitCachedJobRunner
from worker.utils import CompleteJobResult

DECIMALS = 5

INTEGER_DTYPES = ["int8", "int16", "int32", "int64"]
FLOAT_DTYPES = ["float16", "float32", "float64"]


SplitDescriptiveStatsJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class Histogram(TypedDict):
    hist: List[int]
    bin_edges: List[float]


class NumericalStatsItem(TypedDict):
    nan_count: int
    nan_prop: float
    min: float
    max: float
    mean: float
    median: float
    std: float
    histogram: Histogram


class CategoricalStatsItem(TypedDict):
    nan_count: int
    nan_prop: float
    n_unique: int
    frequencies: Dict[str, int]


class StatsPerColumnItem(TypedDict):
    column_name: str
    column_type: str
    column_dtype: Optional[str]
    column_stats: Union[NumericalStatsItem, CategoricalStatsItem]


class SplitDescriptiveStatsResponse(TypedDict):
    num_examples: int
    stats: List[StatsPerColumnItem]


def compute_histogram(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    urls: List[str],
    bin_size: int,
) -> Histogram:
    hist_query = f"""
    SELECT FLOOR("{column_name}"/{bin_size})*{bin_size}, COUNT(*) FILTER (WHERE {column_name} IS NOT NULL)
     FROM read_parquet({urls}) GROUP BY 1 ORDER BY 1;
    """
    logging.debug(f"Compute histogram for {column_name}")
    bins, hist = zip(*con.sql(hist_query).fetchall())  # 2 tuples
    bins, hist = list(bins), list(hist)
    if None in bins:  # if there are None values in column
        # it should be always the last element since we order in query but just in case
        none_idx = bins.index(None)
        bins.pop(none_idx)
        hist.pop(none_idx)
    bins = np.round(bins, DECIMALS).tolist()
    return Histogram(hist=hist, bin_edges=bins)


def compute_numerical_stats(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    urls: List[str],
    n_bins: int,
    n_samples: int,
    dtype: str,
) -> NumericalStatsItem:
    query = f"""
    SELECT min({column_name}), max({column_name}), mean({column_name}), median({column_name}),
     stddev_samp({column_name}) FROM read_parquet({urls});
    """
    minimum, maximum, mean, median, std = duckdb.query(query).fetchall()[0]
    if dtype in FLOAT_DTYPES:
        bin_size = np.round((maximum - minimum) / n_bins, decimals=DECIMALS).item()
        minimum, maximum, mean, median, std = np.round([minimum, maximum, mean, median, std], DECIMALS).tolist()
    elif dtype in INTEGER_DTYPES:
        if maximum - minimum < n_bins:
            bin_size = 1
        else:
            bin_size = int(np.round((maximum - minimum) / n_bins))
        mean, median, std = np.round([mean, median, std], DECIMALS).tolist()
    else:
        raise ValueError("Incorrect dtype, only integers and float are allowed. ")
    nan_query = f"SELECT COUNT(*) FILTER (WHERE {column_name} IS NULL) FROM read_parquet({urls});"
    nan_count = duckdb.query(nan_query).fetchall()[0][0]
    nan_prop = np.round(nan_count / n_samples, DECIMALS).item() if nan_count else 0.0

    histogram = compute_histogram(con, column_name, urls, bin_size=bin_size)
    return NumericalStatsItem(
        nan_count=nan_count,
        nan_prop=nan_prop,
        min=minimum,
        max=maximum,
        mean=mean,
        median=median,
        std=std,
        histogram=histogram,
    )


def compute_categorical_stats(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    urls: List[str],
    class_label_names: List[str],
    n_samples: int,
) -> CategoricalStatsItem:
    query = f"""
    SELECT {column_name}, COUNT(*) from read_parquet({urls}) GROUP BY {column_name};
    """
    categories: List[Tuple[int, int]] = con.sql(query).fetchall()  # list of tuples (idx, num_samples)

    logging.debug(f"Statistics for {column_name} computed")
    frequencies, nan_count = {}, 0
    for cat_id, freq in categories:
        if cat_id is not None:
            frequencies[class_label_names[cat_id]] = freq
        else:
            nan_count = freq
    nan_prop = np.round(nan_count / n_samples, DECIMALS).item() if nan_count != 0 else 0.0
    return CategoricalStatsItem(
        nan_count=nan_count,
        nan_prop=nan_prop,
        n_unique=len(categories),
        frequencies=frequencies,
    )


def compute_descriptive_stats_response(
    dataset: str,
    config: str,
    split: str,
    histogram_num_bins: int,
    max_parquet_size_bytes: int,
) -> SplitDescriptiveStatsResponse:
    con = duckdb.connect()
    con.sql("INSTALL httpfs")
    con.sql("LOAD httpfs")
    con.sql("SET enable_progress_bar=true;")
    logging.info(f"Compute descriptive statistics for {dataset=}, {config=}, {split=}")

    previous_step = "config-parquet-and-info"
    parquet_and_info_best_response = get_previous_step_or_raise(
        kinds=[previous_step],
        dataset=dataset,
        config=config,
    )
    content_parquet_and_info = parquet_and_info_best_response.response["content"]
    if "parquet_files" not in content_parquet_and_info:
        raise PreviousStepFormatError(f"previous step '{previous_step} doesn't return expected field: 'parquet_files'")

    if "dataset_info" not in content_parquet_and_info:
        raise PreviousStepFormatError(f"previous step '{previous_step} doesn't return expected field: 'dataset_info'")

    stats: List[StatsPerColumnItem] = []
    split_parquet_files = [
        parquet_file
        for parquet_file in content_parquet_and_info["parquet_files"]
        if parquet_file["config"] == config and parquet_file["split"] == split
    ]

    split_parquets_size = sum(parquet_file["size"] for parquet_file in split_parquet_files)

    if split_parquets_size > max_parquet_size_bytes:
        # TODO: raise libcommon.exceptions.SplitWithTooBigParquetError after duckdb-index PR is merged
        raise ValueError(
            f"Statistics computation is limited to split parquets under {max_parquet_size_bytes} bytes. "
            f"Current size of sum of split parquets is {split_parquets_size} bytes."
        )

    parquet_files_urls = [parquet_file["url"] for parquet_file in split_parquet_files]

    num_examples = content_parquet_and_info["dataset_info"]["splits"][split]["num_examples"]
    features = content_parquet_and_info["dataset_info"].get("features", [])
    categorical_features = {
        feature_name: feature for feature_name, feature in features.items() if feature.get("_type") == "ClassLabel"
    }

    # compute for ClassLabels (we are sure that these are discrete categories)
    if categorical_features:
        logging.info("Compute statistics for categorical features")
    for feature_name, feature in tqdm(categorical_features.items()):
        logging.info(f"Compute statistics for ClassLabel feature {feature_name}")
        class_label_names = feature["names"]
        cat_column_stats: CategoricalStatsItem = compute_categorical_stats(
            con, feature_name, class_label_names=class_label_names, n_samples=num_examples, urls=parquet_files_urls
        )
        stats.append(
            StatsPerColumnItem(
                column_name=feature_name,
                column_type="class_label",
                column_dtype=None,  # should be some int?
                column_stats=cat_column_stats,
            )
        )

    numerical_columns = {
        feature_name: feature
        for feature_name, feature in features.items()
        if feature.get("_type") == "Value" and feature.get("dtype") in INTEGER_DTYPES + FLOAT_DTYPES
    }
    if numerical_columns:
        logging.info("Compute min, max, mean, median, std, histogram for numerical features. ")
    for feature_name, feature in tqdm(numerical_columns.items()):
        feature_dtype = feature["dtype"]
        num_column_stats: NumericalStatsItem = compute_numerical_stats(
            con,
            feature_name,
            urls=parquet_files_urls,
            n_bins=histogram_num_bins,
            n_samples=num_examples,
            dtype=feature_dtype,
        )
        stats.append(
            StatsPerColumnItem(
                column_name=feature_name,
                column_type="float" if feature_dtype in FLOAT_DTYPES else "int",
                column_dtype=feature_dtype,
                column_stats=num_column_stats,
            )
        )

    return SplitDescriptiveStatsResponse(num_examples=num_examples, stats=stats)


class SplitDescriptiveStatsJobRunner(SplitCachedJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
    ):
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.descriptive_stats_config = app_config.descriptive_stats

    @staticmethod
    def get_job_type() -> str:
        return "split-descriptive-stats"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_DESCRIPTIVE_STATS_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_descriptive_stats_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                histogram_num_bins=self.descriptive_stats_config.histogram_num_bins,
                max_parquet_size_bytes=self.descriptive_stats_config.max_parquet_size_bytes,
            )
        )
