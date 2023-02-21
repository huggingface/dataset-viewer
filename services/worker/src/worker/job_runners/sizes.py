# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional, TypedDict

from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import JobRunner, JobRunnerError

SizesJobRunnerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
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


class SizesContent(TypedDict):
    dataset: DatasetSize
    configs: list[ConfigSize]
    splits: list[SplitSize]


class SizesResponse(TypedDict):
    sizes: SizesContent


class SizesJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SizesJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepStatusError(SizesJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(SizesJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_sizes_response(dataset: str) -> SizesResponse:
    """
    Get the response of /sizes for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `SizesResponse`: An object with the sizes_response.
    <Tip>
    Raises the following errors:
        - [`~job_runners.sizes.PreviousStepStatusError`]
          If the previous step gave an error.
        - [`~job_runners.sizes.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get sizes for dataset={dataset}")

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
    try:
        split_sizes: list[SplitSize] = []
        config_sizes: list[ConfigSize] = []
        for config in content["dataset_info"].keys():
            config_dataset_info = content["dataset_info"][config]
            num_columns = len(config_dataset_info["features"])
            config_split_sizes: list[SplitSize] = [
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
            config_sizes.append(
                {
                    "dataset": dataset,
                    "config": config,
                    "num_bytes_original_files": config_dataset_info["download_size"],
                    "num_bytes_parquet_files": sum(
                        split_size["num_bytes_parquet_files"] for split_size in config_split_sizes
                    ),
                    "num_bytes_memory": sum(
                        split_size["num_bytes_memory"] for split_size in config_split_sizes
                    ),  # or "num_bytes_memory": config_dataset_info["dataset_size"],
                    "num_rows": sum(split_size["num_rows"] for split_size in config_split_sizes),
                    "num_columns": len(config_dataset_info["features"]),
                }
            )
            split_sizes.extend(config_split_sizes)
        dataset_size: DatasetSize = {
            "dataset": dataset,
            "num_bytes_original_files": sum(config_size["num_bytes_original_files"] for config_size in config_sizes),
            "num_bytes_parquet_files": sum(config_size["num_bytes_parquet_files"] for config_size in config_sizes),
            "num_bytes_memory": sum(config_size["num_bytes_memory"] for config_size in config_sizes),
            "num_rows": sum(config_size["num_rows"] for config_size in config_sizes),
        }
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return {
        "sizes": {
            "dataset": dataset_size,
            "configs": config_sizes,
            "splits": split_sizes,
        }
    }


class SizesJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/sizes"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def compute(self) -> Mapping[str, Any]:
        return compute_sizes_response(dataset=self.dataset)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=split_size["dataset"], config=split_size["config"], split=split_size["split"])
            for split_size in content["sizes"]["splits"]
        }
