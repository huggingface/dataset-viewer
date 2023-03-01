# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner

SplitNamesFromDatasetInfoJobRunnerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
]


class SplitNamesFromDatasetInfoJobRunnerError(JobRunnerError):
    """Base class for split names job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SplitNamesFromDatasetInfoJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepStatusError(SplitNamesFromDatasetInfoJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(SplitNamesFromDatasetInfoJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class SplitNameItem(TypedDict):
    dataset: str
    config: str
    split: str


class SplitNamesFromDatasetInfoResponseContent(TypedDict):
    splits: List[SplitNameItem]


def compute_split_names_from_dataset_info_response(
    dataset: str,
    config: str,
) -> SplitNamesFromDatasetInfoResponseContent:
    """
    Get the response of /split-names-from-dataset-info for one specific dataset and config on huggingface.co
    computed from cached response in /dataset-info step.

    The /split-names-from-dataset-info response generated by this function does not include stats about the split,
    like the size or number of samples. See /dataset-info or /sizes for that.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `SplitNamesFromDatasetInfoResponseContent`: An object with the list of split names for the dataset and config.
    <Tip>
    Raises the following errors:
        - [`~job_runners.split_names_from_dataset_info.PreviousStepStatusError`]
          If the the previous step gave an error.
        - [`~job_runners.split_names_from_dataset_info.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
        - [`~libcommon.dataset.DatasetNotFoundError`]
            If previous step content was not found for the dataset
    </Tip>
    """
    logging.info(f"get split names from dataset info for dataset={dataset}, config={config}")
    try:
        response = get_response(kind="/dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError("No response found in previous step for this dataset.", e) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {response['http_status']}. This job should not have been created."
        )

    try:
        splits_content = response["content"]["dataset_info"][config]["splits"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    split_name_items: List[SplitNameItem] = [
        {"dataset": dataset, "config": config, "split": str(split)} for split in splits_content
    ]

    return {"splits": split_name_items}


class SplitNamesFromDatasetInfoJobRunner(DatasetsBasedJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/split-names-from-dataset-info"

    @staticmethod
    def get_version() -> str:
        return "2.0.0"

    def compute(self) -> Mapping[str, Any]:
        if self.dataset is None:
            raise ValueError("dataset is required")
        if self.config is None:
            raise ValueError("config is required")
        return compute_split_names_from_dataset_info_response(dataset=self.dataset, config=self.config)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {SplitFullName(dataset=s["dataset"], config=s["config"], split=s["split"]) for s in content["splits"]}
