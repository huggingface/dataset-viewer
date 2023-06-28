# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Tuple

from libcommon.constants import PROCESSING_STEP_DATASET_SIZE_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CacheEntryDoesNotExistError,
    get_previous_step_or_raise,
    get_response,
)

from worker.dtos import (
    ConfigSize,
    ConfigSizeResponse,
    DatasetSize,
    DatasetSizeResponse,
    JobResult,
    PreviousJob,
    SplitSize,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_sizes_response(dataset: str) -> Tuple[DatasetSizeResponse, float]:
    """
    Get the response of dataset-size for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `DatasetSizeResponse`: An object with the sizes_response.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    """
    logging.info(f"get sizes for dataset={dataset}")

    config_names_best_response = get_previous_step_or_raise(kinds=["dataset-config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    try:
        split_sizes: list[SplitSize] = []
        config_sizes: list[ConfigSize] = []
        total = 0
        pending = []
        failed = []
        for config_item in content["config_names"]:
            config = config_item["config"]
            total += 1
            try:
                response = get_response(kind="config-size", dataset=dataset, config=config)
            except CacheEntryDoesNotExistError:
                logging.debug("No response found in previous step for this dataset: 'config-size' endpoint.")
                pending.append(
                    PreviousJob(
                        {
                            "kind": "config-size",
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
                            "kind": "config-size",
                            "dataset": dataset,
                            "config": config,
                            "split": None,
                        }
                    )
                )
                continue
            config_size_content = ConfigSizeResponse(size=response["content"]["size"])
            config_sizes.append(config_size_content["size"]["config"])
            split_sizes.extend(config_size_content["size"]["splits"])
        dataset_size: DatasetSize = {
            "dataset": dataset,
            "num_bytes_original_files": sum(config_size["num_bytes_original_files"] for config_size in config_sizes),
            "num_bytes_parquet_files": sum(config_size["num_bytes_parquet_files"] for config_size in config_sizes),
            "num_bytes_memory": sum(config_size["num_bytes_memory"] for config_size in config_sizes),
            "num_rows": sum(config_size["num_rows"] for config_size in config_sizes),
        }
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return (
        DatasetSizeResponse(
            {
                "size": {
                    "dataset": dataset_size,
                    "configs": config_sizes,
                    "splits": split_sizes,
                },
                "pending": pending,
                "failed": failed,
            }
        ),
        progress,
    )


class DatasetSizeJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-size"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_SIZE_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_sizes_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
