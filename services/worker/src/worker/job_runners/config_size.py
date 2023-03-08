# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import JobRunner, JobRunnerError

ConfigSizeJobRunnerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "MissingInfoForConfigError",
]


class DatasetSize(TypedDict):
    dataset: str
    num_bytes_original_files: int
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int


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


class PreviousStepStatusError(ConfigSizeJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(ConfigSizeJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class MissingInfoForConfigError(ConfigSizeJobRunnerError):
    """Raised when the dataset info from the parquet export is missing the requested dataset configuration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingInfoForConfigError", cause, False)


def compute_config_size_response(dataset: str, config: str) -> ConfigSizeResponse:
    """
    Get the response of /sizes for one specific dataset and config on huggingface.co.
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
        - [`~job_runners.config_size.PreviousStepStatusError`]
          If the previous step gave an error.
        - [`~job_runners.config_size.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get size for dataset={dataset}, config={config}")

    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError(
            "No response found in previous step for this dataset: '/parquet-and-dataset-info' endpoint.", e
        ) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {response['http_status']}. This job should not have been created."
        )
    content = response["content"]
    if config not in content["dataset_info"]:
        raise MissingInfoForConfigError(
            f"Dataset configuration '{config}' is missing in the dataset info from the parquet export. "
            f"Available configurations: {', '.join(content['dataset_info'][:10])}"
            + f"... ({len(content['dataset_info']) - 10})" if len(content['dataset_info']) > 10 else ""
        )
    try:
        config_dataset_info = content["dataset_info"][config]
        num_columns = len(config_dataset_info["features"])
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
            for split_info in config_dataset_info["splits"].values()
        ]
        config_size = ConfigSize(
            {
                "dataset": dataset,
                "config": config,
                "num_bytes_original_files": config_dataset_info["download_size"],
                "num_bytes_parquet_files": sum(
                    split_size["num_bytes_parquet_files"] for split_size in split_sizes
                ),
                "num_bytes_memory": sum(
                    split_size["num_bytes_memory"] for split_size in split_sizes
                ),  # or "num_bytes_memory": config_dataset_info["dataset_size"],
                "num_rows": sum(split_size["num_rows"] for split_size in split_sizes),
                "num_columns": len(config_dataset_info["features"]),
            }
        )
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return ConfigSizeResponse({
        "size": {
            "config": config_size,
            "splits": split_sizes,
        }
    })


class ConfigSizeJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/config-size"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def compute(self) -> Mapping[str, Any]:
        return compute_config_size_response(dataset=self.dataset, config=self.config)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=split_size["dataset"], config=split_size["config"], split=split_size["split"])
            for split_size in content["sizes"]["splits"]
        }
