# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Dict, List, Literal, Tuple, TypedDict, Union

import duckdb
import polars as pl
from libcommon.constants import PROCESSING_STEP_SPLIT_BASIC_STATS_VERSION
from libcommon.exceptions import PreviousStepFormatError
# from libcommon.exceptions import SplitWithTooBigParquetError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo
from libcommon.simple_cache import get_previous_step_or_raise
from tqdm import tqdm


from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitCachedJobRunner
from worker.utils import CompleteJobResult


SplitBasicStatsJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class Histogram(TypedDict):
    hist: List[int]
    bin_edges: List[float]


class ContinuousStatsItem(TypedDict):
    min: float
    max: float
    mean: float
    median: float
    std: float
    histogram: Histogram


class CategoricalStatsItem(TypedDict):
    n_unique: int
    frequencies: Dict[str, int]


class StatsPerColumnItem(TypedDict):
    column_name: str
    column_type: str
    # column_dtype: str
    column_stats: Union[ContinuousStatsItem, CategoricalStatsItem]


class SplitBasicStatsResponse(TypedDict):
    basic_stats: List[StatsPerColumnItem]


def compute_integer_stats(con, column_name, urls):
    query = f"""
    SELECT min({column_name}), max({column_name}), mean({column_name}), median({column_name}), 
    stddev_samp({column_name}) FROM read_parquet({urls});
    """
    minimum, maximum, mean, median, std = con.sql(query).fetchall()[0]
    if minimum == 0 and maximum < 20:
        pass


def compute_continuous_stats(con, column_name, urls, n_bins):
    query = f"""
    SELECT min({column_name}), max({column_name}), mean({column_name}), median({column_name}), 
    stddev_samp({column_name}) FROM read_parquet({urls});
    """
    minimum, maximum, mean, median, std = duckdb.query(query).fetchall()[0]
    bin_size = (maximum - minimum) / n_bins
    hist_query = f"""
    SELECT FLOOR("{column_name}"/{bin_size})*{bin_size}, COUNT(*) as count 
    FROM read_parquet({urls}) 
    GROUP BY 1 
    ORDER BY 1;
    """
    logging.debug(f"Compute histogram for {column_name}")
    bins, hist = zip(*con.sql(hist_query).fetchall())
    histogram = Histogram(hist=list(hist), bin_edges=list(bins))
    return ContinuousStatsItem(
        min=minimum,
        max=maximum,
        mean=mean,
        median=median,
        std=std,
        histogram=histogram,
    )


def compute_categorical_stats(con, column_name, class_label_names, urls):
    query = f"""
    SELECT {column_name}, COUNT(*) from read_parquet({urls}) GROUP BY {column_name};
    """
    categories: List[Tuple[int, int]] = con.sql(query).fetchall()  # list of tuples (idx, num_samples)
    logging.debug(f"Statistics for {column_name} computed")
    return CategoricalStatsItem(
        n_unique=len(categories), frequencies={class_label_names[cat]: freq for cat, freq in categories},
    )


def compute_basic_stats_response(
    dataset: str,
    config: str,
    split: str,
    histogram_num_bins: int,
    max_parquet_size_bytes: int,
) -> SplitBasicStatsResponse:
    con = duckdb.connect()
    con.sql("INSTALL httpfs")
    con.sql("LOAD httpfs")
    con.sql("SET enable_progress_bar=true;")
    logging.info(f"Compute statistics for {dataset=}, {config=}, {split=}")

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

    # compute for ClassLabels (we are sure that these are discrete categories)
    basic_stats: List[StatsPerColumnItem] = []
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

    features = content_parquet_and_info["dataset_info"].get("features", [])
    categorical_features = {
        feature_name: feature for feature_name, feature in features.items() if feature.get("_type") == "ClassLabel"
    }

    if categorical_features:
        logging.info(f"Compute statistics for categorical features")
    for feature_name, feature in categorical_features.items():
        logging.info(f"Compute statistics for ClassLabel feature {feature_name}")
        class_label_names = feature["names"]
        column_stats = compute_categorical_stats(con, feature_name, class_label_names, parquet_files_urls)
        basic_stats.append(
            StatsPerColumnItem(column_name=feature_name, column_type="class_label", column_stats=column_stats)
        )

    logging.info(f"Compute statistics for numerical features: min, max, mean, median, std, histogram")

    # compute for numerical features  # TODO: maybe consider integers to be categorical? if not many unique values
    df_polars = pl.scan_parquet(parquet_files_urls[0])  # scan first parquet file to get list of numerical cols
    numerical_columns = df_polars.select([pl.col(pl.FLOAT_DTYPES)] + [pl.col(pl.INTEGER_DTYPES)]).columns  # .schema
    for column_name in tqdm(numerical_columns):
        logging.debug(f"Compute statistics for numerical feature {column_name}")
        column_stats = compute_continuous_stats(con, column_name, parquet_files_urls, n_bins=histogram_num_bins)
        basic_stats.append(
            StatsPerColumnItem(column_name=feature_name, column_type="numerical", column_stats=column_stats)
        )

    return SplitBasicStatsResponse(basic_stats=basic_stats)


class SplitBasicStatsJobRunner(SplitCachedJobRunner):
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
        self.basic_stats_config = app_config.basic_stats

    @staticmethod
    def get_job_type() -> str:
        return "split-basic-stats"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_BASIC_STATS_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_basic_stats_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                histogram_num_bins=self.basic_stats_config.histogram_num_bins,
                max_parquet_size_bytes=self.basic_stats_config.max_parquet_size_bytes
            )
        )
