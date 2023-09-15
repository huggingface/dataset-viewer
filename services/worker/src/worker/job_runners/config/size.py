# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_CONFIG_SIZE_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.dtos import CompleteJobResult, ConfigSize, ConfigSizeResponse, SplitSize
from worker.job_runners.config.config_job_runner import ConfigJobRunner


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
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
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
                "num_bytes_memory": split_info["num_bytes"] if "num_bytes" in split_info else 0,
                "num_rows": split_info["num_examples"] if "num_examples" in split_info else 0,
                "num_columns": num_columns,
            }
            for split_info in config_info["splits"].values()
        ]
        config_size = ConfigSize(
            {
                "dataset": dataset,
                "config": config,
                "num_bytes_original_files": config_info.get("download_size"),
                "num_bytes_parquet_files": sum(split_size["num_bytes_parquet_files"] for split_size in split_sizes),
                "num_bytes_memory": sum(split_size["num_bytes_memory"] for split_size in split_sizes),
                "num_rows": sum(split_size["num_rows"] for split_size in split_sizes),
                "num_columns": num_columns,
            }
        )
        partial = content["partial"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return ConfigSizeResponse(
        {
            "size": {
                "config": config_size,
                "splits": split_sizes,
            },
            "partial": partial,
        }
    )


class ConfigSizeJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-size"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_SIZE_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_config_size_response(dataset=self.dataset, config=self.config))
