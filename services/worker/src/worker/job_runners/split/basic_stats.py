# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Dict, List, Literal, TypedDict, Union

import dask.array as da
import dask.dataframe as dd
import duckdb
import polars as pl
from libcommon.constants import PROCESSING_STEP_SPLIT_BASIC_STATS_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitCachedJobRunner
from worker.utils import CompleteJobResult, get_previous_step_or_raise

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
    column_stats: Union[ContinuousStatsItem, CategoricalStatsItem]


class SplitBasicStatsResponse(TypedDict):
    basic_stats: List[StatsPerColumnItem]


def compute_continuous_stats_dask(series: dd.Series) -> ContinuousStatsItem:
    histogram_range = series.min(), series.max()
    hist, bin_edges = dd.compute(*da.histogram(series.to_dask_array(), bins=10, range=histogram_range))
    bin_edges = bin_edges.round(decimals=14)  # round up numpy precision issues
    bin_edges = bin_edges.tolist()
    minimum, maximum = bin_edges[0], bin_edges[-1]
    mean, median, std = series.mean().compute().item(), series.median().compute().item(), series.std().compute().item()
    histogram = Histogram(hist=hist.to_list(), bin_edges=bin_edges)
    return ContinuousStatsItem(
        min=minimum,
        max=maximum,
        mean=mean,
        median=median,
        std=std,
        histogram=histogram,
    )


def compute_continuous_stats(column_name, urls):
    query = f"""
    SELECT min({column_name}), max({column_name}), mean({column_name}), median({column_name}), 
    stddev_samp({column_name}) FROM read_parquet({urls});
    """
    minimum, maximum, mean, median, std = duckdb.query(query).fetchall()[0]
    bin_size = (maximum - minimum) / 10
    hist_query = f"""
    SELECT FLOOR("{column_name}"/{bin_size})*{bin_size}, COUNT(*) as count 
    FROM read_parquet({urls}) 
    GROUP BY 1 
    ORDER BY 1;
    """
    bins, hist = zip(*duckdb.query(hist_query).fetchall())
    histogram = Histogram(hist=list(hist), bin_edges=list(bins))
    return ContinuousStatsItem(
        min=minimum,
        max=maximum,
        mean=mean,
        median=median,
        std=std,
        histogram=histogram,
    )


def compute_categorical_stats(column_name, urls):
    query = f"""
    SELECT {column_name}, COUNT(*) from read_parquet({urls}) GROUP BY {column_name};
    """
    categories = duckdb.query(query).fetchall()
    return CategoricalStatsItem(
        n_unique=len(categories), frequencies=dict(sorted(categories, key=lambda x: x[1], reverse=True))
    )


def compute_basic_stats_response(
    dataset: str,
    config: str,
    split: str,
) -> SplitBasicStatsResponse:
    duckdb.query("INSTALL httpfs")
    duckdb.query("LOAD httpfs")
    duckdb.query("SET enable_progress_bar=true;")
    logging.info(f"get basic statistics for {dataset=}, {config=}, {split=}")

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
    parquet_files_urls = [
        parquet_file["url"]
        for parquet_file in content_parquet_and_info["parquet_files"]
        if parquet_file["config"] == config and parquet_file["split"] == split
    ]

    features = content_parquet_and_info["dataset_info"].get("features", [])
    categorical_columns = [
        feature_name for feature_name, feature in features.items() if feature["_type"] == "ClassLabel"
    ]

    if categorical_columns:
        logging.info(f"Compute statistics for categorical features")
    for column_name in categorical_columns:
        logging.debug(f"Compute statistics for ClassLabel feature {column_name}")
        column_stats = compute_categorical_stats(column_name, parquet_files_urls)
        basic_stats.append(
            StatsPerColumnItem(column_name=column_name, column_type="class_label", column_stats=column_stats)
        )

    logging.info(f"Compute statistics for numerical features")
    # compute for numerical features  # TODO: maybe consider integers to be categorical? if not many unique values
    df_polars = pl.scan_parquet(parquet_files_urls[0])  # scan first parquet file to get list of numerical cols
    numerical_columns = df_polars.select([pl.col(pl.FLOAT_DTYPES)] + [pl.col(pl.INTEGER_DTYPES)]).columns  # .schema
    for column_name in numerical_columns:  # TODO tqdm
        logging.debug(f"Compute statistics for numerical feature {column_name}")
        column_stats = compute_continuous_stats(column_name, parquet_files_urls)
        basic_stats.append(
            StatsPerColumnItem(column_name=column_name, column_type="numerical", column_stats=column_stats)
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

    @staticmethod
    def get_job_type() -> str:
        return "split-basic-stats"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_BASIC_STATS_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_basic_stats_response(dataset=self.dataset, config=self.config, split=self.split)
        )
