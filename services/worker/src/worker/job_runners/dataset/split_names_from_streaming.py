# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict

from libcommon.constants import (
    PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_STREAMING_VERSION,
)
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import (
    JobResult,
    JobRunner,
    JobRunnerError,
    ParameterMissingError,
    get_previous_step_or_raise,
)
from worker.utils import ConfigItem, SplitItem

DatasetSplitNamesFromStreamingJobRunnerErrorCode = Literal["PreviousStepFormatError", "ResponseNotReady"]


class DatasetSplitNamesFromStreamingJobRunnerError(JobRunnerError):
    """Base class for dataset split from streaming names job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetSplitNamesFromStreamingJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(DatasetSplitNamesFromStreamingJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class ResponseNotReadyError(DatasetSplitNamesFromStreamingJobRunnerError):
    """Raised when the response has not been processed yet from any source."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ResponseNotReady")


class FailedConfigItem(ConfigItem):
    error: Mapping[str, Any]


class DatasetSplitNamesFromStreamingResponse(TypedDict):
    splits: List[SplitItem]
    pending: List[ConfigItem]
    failed: List[FailedConfigItem]


def compute_dataset_split_names_from_streaming_response(
    dataset: str,
) -> Tuple[DatasetSplitNamesFromStreamingResponse, float]:
    """
    Get the response of /splits for one specific dataset on huggingface.co
    computed from responses cached in /split-names-from-streaming step.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
    Returns:
        `DatasetSplitNamesFromStreamingResponse`: An object with a list of split names for the dataset [splits],
         a list of pending configs to be processed [pending] and the list of errors [failed] by config.
    <Tip>
    Raises the following errors:
        - [`~job_runner.PreviousStepError`]
          If the the previous step gave an error.
        - [`~job_runners.dataset.split_names_from_streaming.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get dataset split names from dataset info for dataset={dataset}")

    config_names_best_response = get_previous_step_or_raise(kinds=["/config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")
    config_content = content["config_names"]

    try:
        splits: List[SplitItem] = []
        pending: List[ConfigItem] = []
        failed: List[FailedConfigItem] = []
        total = 0
        for config_item in config_content:
            config = config_item["config"]
            total += 1
            try:
                response = get_response(kind="/split-names-from-streaming", dataset=dataset, config=config)
            except DoesNotExist:
                logging.debug("No response found in previous step '/split-names-from-streaming' for this dataset.")
                pending.append(ConfigItem({"dataset": dataset, "config": config}))
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                failed.append(
                    FailedConfigItem(
                        {
                            "dataset": dataset,
                            "config": config,
                            "error": response["content"],
                        }
                    )
                )
                continue
            splits.extend(
                [
                    SplitItem({"dataset": dataset, "config": config, "split": split_content["split"]})
                    for split_content in response["content"]["splits"]
                ]
            )
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return (
        DatasetSplitNamesFromStreamingResponse(
            {
                "splits": splits,
                "pending": pending,
                "failed": failed,
            }
        ),
        progress,
    )


class DatasetSplitNamesFromStreamingJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-split-names-from-streaming"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_STREAMING_VERSION

    def compute(self) -> JobResult:
        if self.dataset is None:
            raise ParameterMissingError("'dataset' parameter is required")
        response_content, progress = compute_dataset_split_names_from_streaming_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {SplitFullName(dataset=s["dataset"], config=s["config"], split=s["split"]) for s in content["splits"]}
