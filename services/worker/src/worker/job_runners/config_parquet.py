# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict

from libcommon.constants import PROCESSING_STEP_PARQUET_VERSION
from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import CompleteJobResult, JobRunner, JobRunnerError
from worker.job_runners.parquet_and_dataset_info import (
    ParquetAndDatasetInfoResponse,
    ParquetFileItem,
)

ConfigParquetJobRunnerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "MissingInfoForConfigError",
]


class ConfigParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]


class ConfigParquetJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigParquetJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepStatusError(ConfigParquetJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(ConfigParquetJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class MissingInfoForConfigError(ConfigParquetJobRunnerError):
    """Raised when the dataset info from the parquet export is missing the requested dataset configuration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingInfoForConfigError", cause, False)


def compute_parquet_response(dataset: str, config: str) -> ConfigParquetResponse:
    """
    Get the response of /parquet for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `ConfigParquetResponse`: An object with the parquet_response (list of parquet files).
    <Tip>
    Raises the following errors:
        - [`~job_runners.parquet.PreviousStepStatusError`]
          If the previous step gave an error.
        - [`~job_runners.parquet.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
        - [`~job_runners.config_size.MissingInfoForConfigError`]
            If the dataset info from the parquet export is missing the requested dataset configuration
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
    </Tip>
    """
    logging.info(f"get parquet files for dataset={dataset}, config={config}")

    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError(
            "No response found in previous step for this dataset: '/parquet-and-dataset-info'.", e
        ) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(f"Previous step gave an error: {response['http_status']}.")
    try:
        content = ParquetAndDatasetInfoResponse(
            parquet_files=response["content"]["parquet_files"], dataset_info=response["content"]["dataset_info"]
        )
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    if config not in content["dataset_info"]:
        if not isinstance(content["dataset_info"], dict):
            raise PreviousStepFormatError(
                "Previous step did not return the expected content.",
                TypeError(f"dataset_info should be a dict, but got {type(content['dataset_info'])}"),
            )
        raise MissingInfoForConfigError(
            f"Dataset configuration '{config}' is missing in the dataset info from the parquet export. "
            f"Available configurations: {', '.join(list(content['dataset_info'])[:10])}"
            + f"... ({len(content['dataset_info']) - 10})"
            if len(content["dataset_info"]) > 10
            else ""
        )

    if "parquet_files" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'parquet_files'.")
    parquet_files = [parquet_file for parquet_file in content["parquet_files"] if parquet_file.get("config") == config]
    return ConfigParquetResponse(parquet_files=parquet_files)


class ConfigParquetJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return 3

    def compute(self) -> CompleteJobResult:
        if self.config is None:
            raise ValueError("config is required")
        return CompleteJobResult(compute_parquet_response(dataset=self.dataset, config=self.config))

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=parquet_file["dataset"], config=parquet_file["config"], split=parquet_file["split"])
            for parquet_file in content["parquet_files"]
        }
