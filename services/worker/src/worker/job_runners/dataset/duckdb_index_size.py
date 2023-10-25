# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus

from libcommon.constants import PROCESSING_STEP_DATASET_DUCKDB_INDEX_SIZE_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CacheEntryDoesNotExistError,
    get_previous_step_or_raise,
    get_response,
)

from worker.dtos import (
    ConfigDuckdbIndexSize,
    ConfigDuckdbIndexSizeResponse,
    DatasetDuckdbIndexSize,
    DatasetDuckdbIndexSizeResponse,
    JobResult,
    PreviousJob,
    SplitDuckdbIndexSize,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_dataset_duckdb_index_size_response(dataset: str) -> tuple[DatasetDuckdbIndexSizeResponse, float]:
    """
    Get the response of config-duckdb-index-size for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `DatasetDuckdbIndexSizeResponse`: An object with the duckdb_index_size_response.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
    """
    logging.info(f"get duckdb_index_size for dataset={dataset}")

    config_names_best_response = get_previous_step_or_raise(kinds=["dataset-config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    try:
        split_duckdb_index_sizes: list[SplitDuckdbIndexSize] = []
        config_duckdb_index_sizes: list[ConfigDuckdbIndexSize] = []
        total = 0
        pending = []
        failed = []
        partial = False
        for config_item in content["config_names"]:
            config = config_item["config"]
            total += 1
            try:
                response = get_response(kind="config-duckdb-index-size", dataset=dataset, config=config)
            except CacheEntryDoesNotExistError:
                logging.debug(
                    "No response found in previous step for this dataset: 'config-duckdb-index-size' endpoint."
                )
                pending.append(
                    PreviousJob(
                        {
                            "kind": "config-duckdb-index-size",
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
                            "kind": "config-duckdb-index-size",
                            "dataset": dataset,
                            "config": config,
                            "split": None,
                        }
                    )
                )
                continue
            config_size_content = ConfigDuckdbIndexSizeResponse(
                size=response["content"]["size"], partial=response["content"]["partial"]
            )
            config_duckdb_index_sizes.append(config_size_content["size"]["config"])
            split_duckdb_index_sizes.extend(config_size_content["size"]["splits"])
            partial = partial or config_size_content["partial"]

        dataset_duckdb_index_size: DatasetDuckdbIndexSize = {
            "dataset": dataset,
            "has_fts": any(
                config_duckdb_index_size["has_fts"] for config_duckdb_index_size in config_duckdb_index_sizes
            ),
            "num_rows": sum(
                config_duckdb_index_size["num_rows"] for config_duckdb_index_size in config_duckdb_index_sizes
            ),
            "num_bytes": sum(
                config_duckdb_index_size["num_bytes"] for config_duckdb_index_size in config_duckdb_index_sizes
            ),
        }
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return (
        DatasetDuckdbIndexSizeResponse(
            {
                "size": {
                    "dataset": dataset_duckdb_index_size,
                    "configs": config_duckdb_index_sizes,
                    "splits": split_duckdb_index_sizes,
                },
                "pending": pending,
                "failed": failed,
                "partial": partial,
            }
        ),
        progress,
    )


class DatasetDuckdbIndexSizeJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-duckdb-index-size"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_DUCKDB_INDEX_SIZE_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_dataset_duckdb_index_size_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
