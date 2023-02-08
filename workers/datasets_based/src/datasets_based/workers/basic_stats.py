# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

import dask.array as da
import dask.dataframe as dd
from datasets_based.worker import Worker
from pandas.api.types import is_numeric_dtype


BasicStatsWorkerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "BasicStatsComputationFailed",
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


class BasicStatsComputationFailed(BasicStatsWorkerError):
    """Raised when the basic stats computation failed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "BasicStatsComputationFailed", cause, False)


def compute_histogram(series = dd.Series) -> Histogram:
    histogram_range = series.min(), series.max()
    hist, bin_edges = dd.compute(*da.histogram(series.to_dask_array(), bins=10, range=histogram_range))
    bin_edges = bin_edges.round(decimals=14)  # round up numpy precision issues
    return Histogram(hist=hist.tolist(), bin_edges=bin_edges.tolist())


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
            If the content of the previous step has not the expected format.
        - [`~workers.sizes.BasicStatsComputationFailed`]
            If the basic stats computation failed.
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
        basic_stats: List[BasicColumnStats] = []
        parquet_files_urls = [
            parquet_file["url"]
            for parquet_file in response["content"]["parquet_files"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]
        df = dd.read_parquet(parquet_files_urls)
        for column_name in df:
            if is_numeric_dtype(df[column_name].dtype):
                histogram = compute_histogram(df[column_name])
                basic_stats.append(BasicColumnStats(
                    dataset=dataset, config=config, split=split, column_name=column_name, histogram=histogram
                ))
    except Exception as e:
        urls_string = str(parquet_files_urls)
        if len(urls_string) > 300:
            urls_string = urls_string[:300] + "..."
        raise BasicStatsComputationFailed(f"Failed to compute stats from {urls_string}", e) from e

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

    def compute(self) -> Mapping[str, Any]:
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return compute_basic_stats_response(dataset=self.dataset, config=self.config, split=self.split)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
