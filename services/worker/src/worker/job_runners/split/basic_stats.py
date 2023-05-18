# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus
from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import dask.dataframe as dd
from libcommon.constants import PROCESSING_STEP_SPLIT_BASIC_STATS_VERSION
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo
from pandas.api.types import is_numeric_dtype

from worker.common_exceptions import JobRunnerError
from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitCachedJobRunner
from worker.utils import CompleteJobResult, get_previous_step_or_raise

SplitBasicStatsJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class SplitBasicStatsJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SplitBasicStatsJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(SplitBasicStatsJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class BasicStatsItem(TypedDict):
    min: Union[int, float]
    max: Union[int, float]
    median: Union[int, float]
    # TODO


class BasicStatsPerColumnItem(TypedDict):
    column_name: str
    column_stats: BasicStatsItem


class SplitBasicStatsResponse(TypedDict):
    basic_stats: List[BasicStatsPerColumnItem]


# def compute_histogram(series = dd.Series) -> Histogram:
#     histogram_range = series.min(), series.max()
#     hist, bin_edges = dd.compute(*da.histogram(series.to_dask_array(), bins=10, range=histogram_range))
#     bin_edges = bin_edges.round(decimals=14)  # round up numpy precision issues
#     return Histogram(hist=hist.tolist(), bin_edges=bin_edges.tolist())


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

    basic_stats: List[BasicStatsPerColumnItem] = []
    parquet_files_urls = [
        parquet_file["url"]
        for parquet_file in content["parquet_files"]
        if parquet_file["config"] == config and parquet_file["split"] == split
    ]
    df = dd.read_parquet(parquet_files_urls)
    for column_name in df:
        if is_numeric_dtype(df[column_name].dtype):
            # this is silly but it's just to try
            minimum = df[column_name].min().compute().item()
            maximum = df[column_name].max().compute().item()
            median = df[column_name].median().compute().item()
            basic_stats.append(
                BasicStatsPerColumnItem(
                    column_name=column_name, column_stats=BasicStatsItem(min=minimum, max=maximum, median=median)
                )
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
