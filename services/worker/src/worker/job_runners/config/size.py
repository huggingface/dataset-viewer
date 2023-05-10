# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Literal, Optional, TypedDict

from libcommon.constants import PROCESSING_STEP_CONFIG_SIZE_VERSION

from worker.job_runner import (
    CompleteJobResult,
    JobRunner,
    JobRunnerError,
    ParameterMissingError,
    get_previous_step_or_raise,
)

ConfigSizeJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class ConfigSize(TypedDict):
    dataset: str
    config: str
    num_bytes_original_files: int
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int


class SplitSize(TypedDict):
    dataset: str
    config: str
    split: str
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int


class ConfigSizeContent(TypedDict):
    config: ConfigSize
    splits: list[SplitSize]


class ConfigSizeResponse(TypedDict):
    size: ConfigSizeContent


class ConfigSizeJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigSizeJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(ConfigSizeJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_config_size_response(dataset: str, config: str) -> ConfigSizeResponse:
    """
    Get the response of config-size for one specific dataset and config on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `ConfigSizeResponse`: An object with the size_response.
    <Tip>
    Raises the following errors:
        - [`~job_runner.PreviousStepError`]
            If the previous step gave an error.
        - [`~job_runners.config.size.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get size for dataset={dataset}, config={config}")

    dataset_info_best_response = get_previous_step_or_raise(
        kinds=["config-parquet-and-info"], dataset=dataset, config=config
    )
    content = dataset_info_best_response.response["content"]
    if "dataset_info" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.")
    if not isinstance(content["dataset_info"], dict):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"dataset_info should be a dict, but got {type(content['dataset_info'])}"),
        )

    try:
        config_info = content["dataset_info"]
        num_columns = len(config_info["features"])
        split_sizes: list[SplitSize] = [
            {
                "dataset": dataset,
                "config": config,
                "split": split_info["name"],
                "num_bytes_parquet_files": sum(
                    x["size"]
                    for x in content["parquet_files"]
                    if x["config"] == config and x["split"] == split_info["name"]
                ),
                "num_bytes_memory": split_info["num_bytes"],
                "num_rows": split_info["num_examples"],
                "num_columns": num_columns,
            }
            for split_info in config_info["splits"].values()
        ]
        config_size = ConfigSize(
            {
                "dataset": dataset,
                "config": config,
                "num_bytes_original_files": config_info["download_size"],
                "num_bytes_parquet_files": sum(split_size["num_bytes_parquet_files"] for split_size in split_sizes),
                "num_bytes_memory": sum(
                    split_size["num_bytes_memory"] for split_size in split_sizes
                ),  # or "num_bytes_memory": config_dataset_info["dataset_size"],
                "num_rows": sum(split_size["num_rows"] for split_size in split_sizes),
                "num_columns": num_columns,
            }
        )
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return ConfigSizeResponse(
        {
            "size": {
                "config": config_size,
                "splits": split_sizes,
            }
        }
    )


class ConfigSizeJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-size"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_SIZE_VERSION

    def compute(self) -> CompleteJobResult:
        if self.dataset is None:
            raise ParameterMissingError("'dataset' parameter is required")
        if self.config is None:
            raise ParameterMissingError("'config' parameter is required")
        return CompleteJobResult(compute_config_size_response(dataset=self.dataset, config=self.config))
