# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from datasets_based.config import AppConfig
from datasets_based.worker import JobInfo, Worker
from datasets_based.workers.parquet_and_dataset_info import ParquetFileItem

ParquetWorkerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
]


class ParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]


class ParquetWorkerError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ParquetWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class PreviousStepStatusError(ParquetWorkerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(ParquetWorkerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_parquet_response(dataset: str) -> ParquetResponse:
    """
    Get the response of /parquet for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `ParquetResponse`: An object with the parquet_response (list of parquet files).
    <Tip>
    Raises the following errors:
        - [`~workers.parquet.PreviousStepStatusError`]
          If the previous step gave an error.
        - [`~workers.parquet.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get parquet files for dataset={dataset}")

    # TODO: we should move this dependency to the Worker class: defining which are the inputs, and just getting their
    # value here
    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError(
            "No response found in previous step for this dataset: '/parquet-and-dataset-info' endpoint.", e
        ) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {response['http_status']}. This job should not have been created."
        )
    content = response["content"]
    if "parquet_files" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'parquet_files'.")
    return {
        "parquet_files": content["parquet_files"],
    }


class ParquetWorker(Worker):
    @staticmethod
    def get_job_type() -> str:
        return "/parquet"

    @staticmethod
    def get_version() -> str:
        return "3.0.0"

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
        return compute_parquet_response(dataset=self.dataset)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=parquet_file["dataset"], config=parquet_file["config"], split=parquet_file["split"])
            for parquet_file in content["parquet_files"]
        }
