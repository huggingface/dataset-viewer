# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

import dask.dataframe as dd
import numpy as np
from datasets_based.config import AppConfig
from datasets_based.worker import JobInfo, Worker


BasicStatsWorkerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
]

class Histogram(TypedDict):
    hist: List[int]
    bin_edges: List[float]


class BasicColumnStats(TypedDict):
    dataset: str
    config: str
    split: str
    column_name: str
    histogram: Optional[Histogram]
    # TODO: add more basic stats


class BasicStatsResponse(TypedDict):
    basic_stats: List[BasicColumnStats]


class BasicStatsWorkerError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: BasicStatsWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class PreviousStepStatusError(BasicStatsWorkerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(BasicStatsWorkerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_basic_stats_response(dataset: str, config: str, split: str) -> BasicStatsResponse:
    """
    Get the response of /basic-stats for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
    Returns:
        `BasicStatsResponse`: An object with the basic_stats.
    <Tip>
    Raises the following errors:
        - [`~workers.sizes.PreviousStepStatusError`]
          If the the previous step gave an error.
        - [`~workers.sizes.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get basic-stats for dataset={dataset}")

    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError("No response found in previous step for this dataset.", e) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {response['http_status']}. This job should not have been created."
        )
    content = response["content"]
    try:
        basic_stats = []
        parquet_files_urls = [
            parquet_file["url"]
            for parquet_file in response["content"]["parquet_files"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]
        df = dd.read_parquet(parquet_files_urls)
        for col in df:
            if isinstance(df[col].dtype, np.number):
                hist, bin_edges = np.histogram(df[col])
                histogram = Histogram(hist=hist.tolist(), bin_edges=bin_edges.tolist())
                basic_stats.append(BasicColumnStats(
                    dataset, config, split, col, histogram=histogram
                ))
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    basic_stats = BasicStatsResponse(basic_stats)
    return {
        "basic_stats": basic_stats
    }


class BasicStatsWorker(Worker):
    @staticmethod
    def get_job_type() -> str:
        return "/basic-stats"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def __init__(self, job_info: JobInfo, app_config: AppConfig) -> None:
        job_type = job_info["type"]
        try:
            processing_step = app_config.processing_graph.graph.get_step_by_job_type(job_type)
        except ValueError as e:
            raise ValueError(
                f"Unsupported job type: '{job_type}'. The job types declared in the processing graph are:"
                f" {[step.job_type for step in app_config.processing_graph.graph.steps.values()]}"
            ) from e
        super().__init__(job_info=job_info, common_config=app_config.common, processing_step=processing_step)

    def compute(self) -> Mapping[str, Any]:
        return compute_basic_stats_response(dataset=self.dataset, config=self.config, split=self.split)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
