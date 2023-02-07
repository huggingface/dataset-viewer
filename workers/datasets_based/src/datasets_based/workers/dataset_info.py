# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from datasets_based.worker import Worker, WorkerError

DatasetInfoWorkerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
]


class DatasetInfoResponse(TypedDict):
    dataset_info: dict[str, Any]


class DatasetInfoWorkerError(WorkerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetInfoWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepStatusError(DatasetInfoWorkerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(DatasetInfoWorkerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_dataset_info_response(dataset: str) -> DatasetInfoResponse:
    """
    Get the response of /dataset-info for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `DatasetInfoResponse`: An object with the dataset_info response.
    <Tip>
    Raises the following errors:
        - [`~workers.dataset_info.PreviousStepStatusError`]
          If the the previous step gave an error.
        - [`~workers.dataset_info.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get dataset_info for dataset={dataset}")

    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError("No response found in previous step for this dataset.", e) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {response['http_status']}. This job should not have been created."
        )
    content = response["content"]
    if "dataset_info" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content.")
    return {
        "dataset_info": content["dataset_info"],
    }


class DatasetInfoWorker(Worker):
    @staticmethod
    def get_job_type() -> str:
        return "/dataset-info"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def compute(self) -> Mapping[str, Any]:
        return compute_dataset_info_response(dataset=self.dataset)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=self.dataset, config=config, split=split)
            for config in content["dataset_info"].keys()
            for split in content["dataset_info"][config]["splits"].keys()
        }
