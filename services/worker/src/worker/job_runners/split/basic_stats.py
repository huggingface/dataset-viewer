# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import List, Literal, TypedDict

import dask.array as da
import dask.dataframe as dd
from libcommon.constants import PROCESSING_STEP_SPLIT_BASIC_STATS_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo
from pandas.api.types import is_numeric_dtype

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


class ContinuousStatsPerColumnItem(TypedDict):
    column_name: str
    # column_type: str
    column_stats: ContinuousStatsItem


class SplitBasicStatsResponse(TypedDict):
    basic_stats: List[ContinuousStatsPerColumnItem]


def compute_continuous_stats(series: dd.Series) -> ContinuousStatsItem:
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


def compute_basic_stats_response(
    dataset: str,
    config: str,
    split: str,
) -> SplitBasicStatsResponse:
    logging.info(f"get basic statistics for {dataset=}, {config=}, {split=}")

    previous_step = "config-parquet"
    config_parquet_best_response = get_previous_step_or_raise(
        kinds=[previous_step],
        dataset=dataset,
        config=config,
    )
    content = config_parquet_best_response.response["content"]
    if "parquet_files" not in content:
        raise PreviousStepFormatError(f"previous step '{previous_step} doesn't return expected field: 'parquet_files'")

    basic_stats: List[ContinuousStatsPerColumnItem] = []
    parquet_files_urls = [
        parquet_file["url"]
        for parquet_file in content["parquet_files"]
        if parquet_file["config"] == config and parquet_file["split"] == split
    ]
    df = dd.read_parquet(parquet_files_urls)
    for column_name in df:
        if is_numeric_dtype(df[column_name].dtype):
            column_stats = compute_continuous_stats(df[column_name])
            basic_stats.append(ContinuousStatsPerColumnItem(column_name=column_name, column_stats=column_stats))

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
