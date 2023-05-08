# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict

from libcommon.constants import PROCESSING_STEP_DATASET_PARQUET_VERSION
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_operators.config.parquet import ConfigParquetResponse
from worker.job_operators.config.parquet_and_info import ParquetFileItem
from worker.job_operators.dataset.dataset_job_operator import DatasetJobOperator
from worker.job_runner import (
    JobRunnerError,
    get_previous_step_or_raise,
)
from worker.utils import JobResult, PreviousJob

SizesJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class DatasetParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]
    pending: list[PreviousJob]
    failed: list[PreviousJob]


class DatasetParquetJobRunnerError(JobRunnerError):
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


class PreviousStepFormatError(DatasetParquetJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_sizes_response(dataset: str) -> Tuple[DatasetParquetResponse, float]:
    """
    Get the response of dataset-parquet for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `DatasetParquetResponse`: An object with the parquet_response (list of parquet files).
    <Tip>
    Raises the following errors:
        - [`~job_runner.PreviousStepError`]
          If the previous step gave an error.
        - [`~job_runners.dataset.parquet.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get parquet files for dataset={dataset}")

    config_names_best_response = get_previous_step_or_raise(kinds=["/config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    try:
        parquet_files: list[ParquetFileItem] = []
        total = 0
        pending = []
        failed = []
        for config_item in content["config_names"]:
            config = config_item["config"]
            total += 1
            try:
                response = get_response(kind="config-parquet", dataset=dataset, config=config)
            except DoesNotExist:
                logging.debug("No response found in previous step for this dataset: 'config-parquet' endpoint.")
                pending.append(
                    PreviousJob(
                        {
                            "kind": "config-parquet",
                            "dataset": dataset,
                            "config": config,
                            "split": None,
                        }
                    )
                )
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                failed.append(
                    PreviousJob(
                        {
                            "kind": "config-parquet",
                            "dataset": dataset,
                            "config": config,
                            "split": None,
                        }
                    )
                )
                continue
            config_parquet_content = ConfigParquetResponse(parquet_files=response["content"]["parquet_files"])
            parquet_files.extend(config_parquet_content["parquet_files"])
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return (
        DatasetParquetResponse(
            parquet_files=parquet_files,
            pending=pending,
            failed=failed,
        ),
        progress,
    )


class DatasetParquetJobOperator(DatasetJobOperator):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_PARQUET_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_sizes_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=parquet_file["dataset"], config=parquet_file["config"], split=parquet_file["split"])
            for parquet_file in content["parquet_files"]
        }
