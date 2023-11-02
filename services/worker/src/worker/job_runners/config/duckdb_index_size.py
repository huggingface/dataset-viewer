# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus

from libcommon.constants import PROCESSING_STEP_CONFIG_DUCKDB_INDEX_SIZE_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CacheEntryDoesNotExistError,
    get_previous_step_or_raise,
    get_response,
)

from worker.dtos import (
    CompleteJobResult,
    ConfigDuckdbIndexSize,
    ConfigDuckdbIndexSizeResponse,
    SplitDuckdbIndexSize,
)
from worker.job_runners.config.config_job_runner import ConfigJobRunner


def compute_config_duckdb_index_size_response(dataset: str, config: str) -> ConfigDuckdbIndexSizeResponse:
    """
    Get the response of config-duckdb-index-size for one specific dataset and config on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `ConfigDuckdbIndexSizeResponse`: An object with the duckdb_index_size_response.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
    """
    logging.info(f"get duckdb_index_size for dataset={dataset}, config={config}")

    split_names_response = get_previous_step_or_raise(
        kinds=["config-split-names-from-streaming", "config-split-names-from-info"],
        dataset=dataset,
        config=config,
    )
    content = split_names_response.response["content"]
    if "splits" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    try:
        total = 0
        split_duckdb_index_sizes: list[SplitDuckdbIndexSize] = []
        partial = False
        for split_item in content["splits"]:
            split = split_item["split"]
            total += 1
            try:
                duckdb_index_response = get_response(
                    kind="split-duckdb-index", dataset=dataset, config=config, split=split
                )
                config_info_response = get_response(kind="config-info", dataset=dataset, config=config)
            except CacheEntryDoesNotExistError:
                logging.debug(
                    "No response found in previous step for this dataset: 'split-duckdb-index' or 'config-info'."
                )
                continue
            if duckdb_index_response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {duckdb_index_response['http_status']}.")
                continue
            if config_info_response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {config_info_response['http_status']}.")
                continue

            split_duckdb_index = duckdb_index_response["content"]
            config_info = config_info_response["content"]
            if (
                "num_rows" in split_duckdb_index
                and isinstance(split_duckdb_index["num_rows"], int)
                and "num_bytes" in split_duckdb_index
                and isinstance(split_duckdb_index["num_bytes"], int)
            ):
                split_duckdb_index_sizes.append(
                    SplitDuckdbIndexSize(
                        dataset=dataset,
                        config=config,
                        split=split,
                        has_fts=split_duckdb_index["has_fts"],
                        num_rows=split_duckdb_index["num_rows"],
                        num_bytes=split_duckdb_index["num_bytes"],
                    )
                )
                partial = partial or split_duckdb_index["partial"]
            else:
                split_info = config_info["dataset_info"]["splits"][split]
                split_duckdb_index_sizes.append(
                    SplitDuckdbIndexSize(
                        dataset=dataset,
                        config=config,
                        split=split,
                        has_fts=split_duckdb_index["has_fts"],
                        num_rows=split_info["num_rows"],
                        num_bytes=split_info["num_examples"],
                    )
                )
                partial = partial or config_info["partial"]

    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    config_duckdb_index_size = ConfigDuckdbIndexSize(
        dataset=dataset,
        config=config,
        has_fts=any(split_duckdb_index_size["has_fts"] for split_duckdb_index_size in split_duckdb_index_sizes),
        num_rows=sum(split_duckdb_index_size["num_rows"] for split_duckdb_index_size in split_duckdb_index_sizes),
        num_bytes=sum(split_duckdb_index_size["num_bytes"] for split_duckdb_index_size in split_duckdb_index_sizes),
    )

    return ConfigDuckdbIndexSizeResponse(
        {
            "size": {
                "config": config_duckdb_index_size,
                "splits": split_duckdb_index_sizes,
            },
            "partial": partial,
        }
    )


class ConfigDuckdbIndexSizeJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-duckdb-index-size"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_DUCKDB_INDEX_SIZE_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_config_duckdb_index_size_response(dataset=self.dataset, config=self.config))
