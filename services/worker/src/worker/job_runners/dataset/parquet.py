# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import List, Tuple, TypedDict

from libcommon.constants import PROCESSING_STEP_DATASET_PARQUET_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import DoesNotExist, get_response

from worker.job_runners.config.parquet import ConfigParquetResponse
from worker.job_runners.config.parquet_and_info import ParquetFileItem
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import JobResult, PreviousJob, get_previous_step_or_raise


class DatasetParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]
    pending: list[PreviousJob]
    failed: list[PreviousJob]


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


class DatasetParquetJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_PARQUET_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_sizes_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
