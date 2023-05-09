# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, Tuple

from libcommon.constants import PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION
from libcommon.simple_cache import SplitFullName, get_best_response

from worker.common_exceptions import JobRunnerError
from worker.job_operators.dataset.dataset_job_operator import DatasetJobOperator
from worker.utils import (
    ConfigItem,
    DatasetSplitNamesResponse,
    FailedConfigItem,
    JobResult,
    SplitItem,
    get_previous_step_or_raise,
)

DatasetSplitNamesErrorCode = Literal["PreviousStepFormatError"]


class DatasetSplitNamesJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetSplitNamesErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(DatasetSplitNamesJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_dataset_split_names_response(dataset: str) -> Tuple[DatasetSplitNamesResponse, float]:
    """
    Get the response of /splits for one specific dataset on huggingface.co
    computed from responses cached in /split-names-from-dataset-info or /split-names-from-streaming steps.
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
        - [`~job_runners.dataset_split_names.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get dataset split names for dataset={dataset}")

    # Get the config names from the previous steps
    config_names_best_response = get_previous_step_or_raise(kinds=["/config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("'/config-names' did not return the expected content: 'config_names'.")
    config_names = [config_name_item["config"] for config_name_item in content["config_names"]]
    if any(not isinstance(config_name, str) for config_name in config_names):
        raise PreviousStepFormatError("Previous step '/config-names' did not return a list of config names.")

    split_names_cache_kinds = ["/split-names-from-dataset-info", "/split-names-from-streaming"]
    try:
        splits: List[SplitItem] = []
        pending: List[ConfigItem] = []
        failed: List[FailedConfigItem] = []
        total = 0
        for config in config_names:
            total += 1
            best_response = get_best_response(split_names_cache_kinds, dataset=dataset, config=config)
            if best_response.response["error_code"] == "CachedResponseNotFound":
                logging.debug(
                    "No response (successful or erroneous) found in cache for the previous steps"
                    f" '{split_names_cache_kinds}' for this dataset."
                )
                pending.append(ConfigItem({"dataset": dataset, "config": config}))
                continue
            if best_response.response["http_status"] != HTTPStatus.OK:
                logging.debug(f"No successful response found in the previous steps {split_names_cache_kinds}.")
                failed.append(
                    FailedConfigItem(
                        {
                            "dataset": dataset,
                            "config": config,
                            "error": best_response.response["content"],
                        }
                    )
                )
                continue
            splits.extend(
                [
                    SplitItem({"dataset": dataset, "config": config, "split": split_content["split"]})
                    for split_content in best_response.response["content"]["splits"]
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


class DatasetSplitNamesJobOperator(DatasetJobOperator):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-split-names"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_dataset_split_names_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=split_item["dataset"], config=split_item["config"], split=split_item["split"])
            for split_item in content["splits"]
        }
