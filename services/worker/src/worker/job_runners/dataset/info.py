# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from libcommon.constants import PROCESSING_STEP_DATASET_INFO_VERSION
from libcommon.simple_cache import DoesNotExist, get_response

from worker.common_exceptions import JobRunnerError
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import JobResult, PreviousJob, get_previous_step_or_raise

DatasetInfoJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class DatasetInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]
    pending: List[PreviousJob]
    failed: List[PreviousJob]


class DatasetInfoJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetInfoJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(DatasetInfoJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_dataset_info_response(dataset: str) -> Tuple[DatasetInfoResponse, float]:
    """
    Get the response of dataset-info for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        (`DatasetInfoResponse`, `float`): Tuple of an object with the dataset_info response and
        progress float value from 0. to 1. which corresponds to the percentage of dataset configs
        correctly processed and included in current response (some configs might not exist in cache yet
        or raise errors).
    <Tip>
    Raises the following errors:
        - [`~job_runner.PreviousStepError`]
            If the previous step gave an error.
        - [`~job_runners.dataset.info.PreviousStepFormatError`]
            If the content of the previous step doesn't have the expected format.
    </Tip>
    """
    logging.info(f"get dataset_info for {dataset=}")

    config_names_best_response = get_previous_step_or_raise(kinds=["/config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    try:
        config_infos: Dict[str, Any] = {}
        total = 0
        pending, failed = [], []
        for config_item in content["config_names"]:
            config = config_item["config"]
            total += 1
            try:
                config_response = get_response(kind="config-info", dataset=dataset, config=config)
            except DoesNotExist:
                logging.debug(f"No response found in previous step for {dataset=} {config=}: 'config-info'.")
                pending.append(
                    PreviousJob(
                        kind="config-info",
                        dataset=dataset,
                        config=config,
                        split=None,
                    )
                )
                continue
            if config_response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {config_response['http_status']}")
                failed.append(
                    PreviousJob(
                        kind="config-info",
                        dataset=dataset,
                        config=config,
                        split=None,
                    )
                )
                continue
            config_infos[config] = config_response["content"]["dataset_info"]

    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return DatasetInfoResponse(dataset_info=config_infos, pending=pending, failed=failed), progress


class DatasetInfoJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_INFO_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_dataset_info_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
