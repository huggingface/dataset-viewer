# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, Tuple

from libcommon.constants import (
    PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
)
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import (
    JobResult,
    JobRunner,
    JobRunnerError,
    get_previous_step_or_raise,
)
from worker.utils import (
    ConfigItem,
    DatasetSplitNamesResponse,
    FailedConfigItem,
    SplitItem,
)

DatasetSplitNamesFromDatasetInfoErrorCode = Literal["PreviousStepFormatError"]


class DatasetSplitNamesFromDatasetInfoJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetSplitNamesFromDatasetInfoErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(DatasetSplitNamesFromDatasetInfoJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_dataset_split_names_from_dataset_info_response(dataset: str) -> Tuple[DatasetSplitNamesResponse, float]:
    """
    Get the response of /splits for one specific dataset on huggingface.co
    computed from responses cached in /split-names-from-dataset-info step.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
    Returns:
        `DatasetSplitNamesResponse`: An object with a list of split names for the dataset [splits],
         a list of pending configs to be processed [pending] and the list of errors [failed] by config.
    <Tip>
    Raises the following errors:
        - [`~job_runner.PreviousStepError`]
          If the the previous step gave an error.
        - [`~job_runners.dataset.split_names_from_dataset_info.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get dataset split names from dataset info for dataset={dataset}")

    best_dataset_info_response = get_previous_step_or_raise(kinds=["dataset-info"], dataset=dataset)
    content = best_dataset_info_response.response["content"]
    if "dataset_info" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.")
    dataset_info_content = content["dataset_info"]

    try:
        splits: List[SplitItem] = []
        pending: List[ConfigItem] = []
        failed: List[FailedConfigItem] = []
        total = 0
        for config in dataset_info_content.keys():
            total += 1
            try:
                response = get_response(kind="/split-names-from-dataset-info", dataset=dataset, config=config)
            except DoesNotExist:
                logging.debug("No response found in previous step '/split-names-from-dataset-info' for this dataset.")
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
        DatasetSplitNamesResponse(
            {
                "splits": splits,
                "pending": pending,
                "failed": failed,
            }
        ),
        progress,
    )


class DatasetSplitNamesFromDatasetInfoJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-split-names-from-dataset-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_DATASET_INFO_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_dataset_split_names_from_dataset_info_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=split_item["dataset"], config=split_item["config"], split=split_item["split"])
            for split_item in content["splits"]
        }
