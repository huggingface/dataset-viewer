# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import (
    CacheEntryWithInfo,
    DoesNotExist,
    SplitFullName,
    get_response,
    get_responses_for_kind,
)

from worker.job_runner import JobResult, JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner

DatasetSplitNamesJobRunnerErrorCode = Literal["PreviousStepStatusError", "PreviousStepFormatError", "ResponseNotReady"]


class DatasetSplitNamesJobRunnerError(JobRunnerError):
    """Base class for split names job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetSplitNamesJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepStatusError(DatasetSplitNamesJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(DatasetSplitNamesJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class ResponseNotReadyError(DatasetSplitNamesJobRunnerError):
    """Raised when the response has not been processed yet from any source."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ResponseNotReady")


class PendingJob(TypedDict):
    dataset: str
    config: Optional[str]


class SuccessJob(PendingJob):
    split: str


class FailedJob(PendingJob):
    error: Mapping[str, Any]


class DatasetSplitNamesResponseContent(TypedDict):
    splits: List[SuccessJob]
    pending: List[PendingJob]
    failed: List[FailedJob]


class ConfigSource(TypedDict):
    streaming: Optional[CacheEntryWithInfo]
    dataset_info: Optional[CacheEntryWithInfo]


def get_response_from_sources(config_name: str, sources: Dict[str, ConfigSource]) -> CacheEntryWithInfo:
    config_source = sources[config_name]
    result_from_streaming = config_source["streaming"]
    last_result = None
    if result_from_streaming:
        if result_from_streaming["http_status"] == HTTPStatus.OK:
            return result_from_streaming
        last_result = result_from_streaming
    result_from_dataset_info = config_source["dataset_info"]
    if result_from_dataset_info:
        if result_from_dataset_info["http_status"] == HTTPStatus.OK:
            return result_from_dataset_info
        last_result = result_from_dataset_info
    if last_result:
        return last_result

    raise ResponseNotReadyError("Not ready yet")


def compute_dataset_split_names_response(
    dataset: str,
) -> Tuple[DatasetSplitNamesResponseContent, float]:
    """
    Get the response of /splits for one specific dataset on huggingface.co
    computed from responses cached in /split-names-from-streaming and /split-names-from-dataset-info steps.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
    Returns:
        `DatasetSplitNamesResponseContent`: An object with a list of split names for the dataset [splits],
         a list of pending configs to be processed [pending] and the list of errors [failed] by config.
    <Tip>
    Raises the following errors:
        - [`~job_runners.dataset_split_names.PreviousStepStatusError`]
          If the the previous step gave an error.
        - [`~job_runners.dataset_split_names.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
        - [`~libcommon.dataset.DatasetNotFoundError`]
            If previous step content was not found for the dataset
    </Tip>
    """
    logging.info(f"get dataset split names from dataset info for dataset={dataset}")
    try:
        config_names = get_response(kind="/config-names", dataset=dataset)
        config_content = config_names["content"]["config_names"]
    except DoesNotExist as e:
        raise DatasetNotFoundError("No response found in previous step '/config-names' for this dataset.", e) from e
    except KeyError as e:
        raise PreviousStepFormatError("Previous step '/config-names' did not return the expected content.") from e

    if config_names["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {config_names['http_status']}. This job should not have been created."
        )
    try:
        split_names_from_streaming = get_responses_for_kind(kind="/split-names-from-streaming", dataset=dataset)
        split_names_from_dataset_info = get_responses_for_kind(kind="/split-names-from-dataset-info", dataset=dataset)
        streaming_source = {config_item["config"]: config_item for config_item in split_names_from_streaming}
        dataset_info_source = {config_item["config"]: config_item for config_item in split_names_from_dataset_info}

        sources = {
            config_item["config"]: ConfigSource(
                {
                    "streaming": streaming_source.get(config_item["config"]),
                    "dataset_info": dataset_info_source.get(config_item["config"]),
                }
            )
            for config_item in config_content
        }

        splits: List[SuccessJob] = []
        pending: List[PendingJob] = []
        failed: List[FailedJob] = []
        total = 0
        for config in config_content.keys():
            total += 1
            try:
                result = get_response_from_sources(config, sources)
                if result["http_status"] == HTTPStatus.OK:
                    splits.extend(
                        [
                            {"dataset": dataset, "config": config, "split": split_content["split"]}
                            for split_content in result["content"]["splits"]
                        ]
                    )
                else:
                    failed.append({"dataset": dataset, "config": config, "error": result["content"]})
            except ResponseNotReadyError:
                pending.append({"dataset": dataset, "config": config})
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return (
        DatasetSplitNamesResponseContent(
            {
                "splits": splits,
                "pending": pending,
                "failed": failed,
            }
        ),
        progress,
    )


class DatasetSplitNamesJobRunner(DatasetsBasedJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-split-names"

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    def compute(self) -> JobResult:
        response_content, progress = compute_dataset_split_names_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {SplitFullName(dataset=s["dataset"], config=s["config"], split=s["split"]) for s in content["splits"]}
