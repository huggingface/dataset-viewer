# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict

from libcommon.constants import PROCESSING_STEP_CONFIG_PARQUET_VERSION
from libcommon.simple_cache import SplitFullName

from worker.job_runner import (
    CompleteJobResult,
    JobRunnerError,
    get_previous_step_or_raise,
)
from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.job_runners.config.parquet_and_info import ParquetFileItem

ConfigParquetJobRunnerErrorCode = Literal["PreviousStepFormatError"]


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


class PreviousStepFormatError(ConfigParquetJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


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
        - [`~job_runner.PreviousStepError`]
            If the previous step gave an error.
        - [`~job_runners.parquet.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get parquet files for dataset={dataset}, config={config}")

    previous_step = "config-parquet-and-info"
    config_parquet_and_info_best_response = get_previous_step_or_raise(
        kinds=[previous_step], dataset=dataset, config=config
    )
    content = config_parquet_and_info_best_response.response["content"]
    if "parquet_files" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'parquet_files'.")

    parquet_files = [parquet_file for parquet_file in content["parquet_files"] if parquet_file.get("config") == config]
    return ConfigParquetResponse(parquet_files=parquet_files)


class ConfigParquetJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_PARQUET_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_parquet_response(dataset=self.dataset, config=self.config))

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=self.dataset, config=self.config, split=parquet_file["split"])
            for parquet_file in content["parquet_files"]
        }
