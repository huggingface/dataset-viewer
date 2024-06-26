# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.dtos import CompleteJobResult, ConfigSize, ConfigSizeResponse, SplitSize
from worker.job_runners.config.config_job_runner import ConfigJobRunner


def compute_config_size_response(dataset: str, config: str) -> ConfigSizeResponse:
    """
    Get the response of 'config-size' for one specific dataset and config on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format

    Returns:
        `ConfigSizeResponse`: An object with the size_response.
    """
    logging.info(f"compute 'config-size' for {dataset=} {config=}")

    dataset_info_response = get_previous_step_or_raise(kind="config-parquet-and-info", dataset=dataset, config=config)
    content = dataset_info_response["content"]
    if "dataset_info" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.")
    if not isinstance(content["dataset_info"], dict):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"dataset_info should be a dict, but got {type(content['dataset_info'])}"),
        )
    if content["estimated_dataset_info"] is not None and not isinstance(content["estimated_dataset_info"], dict):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"estimated_info should be a dict, but got {type(content['dataset_info'])}"),
        )

    try:
        config_info = content["dataset_info"]
        config_estimated_info = content["estimated_dataset_info"]
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
                "estimated_num_rows": config_estimated_info["splits"][split_info["name"]]["num_examples"]
                if isinstance(config_estimated_info, dict)
                and "splits" in config_estimated_info
                and "name" in split_info
                and split_info["name"] in config_estimated_info["splits"]
                and "num_examples" in config_estimated_info["splits"][split_info["name"]]
                else None,
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
                "estimated_num_rows": sum(
                    split_size["estimated_num_rows"] or split_size["num_rows"] for split_size in split_sizes
                )
                if any(split_size["estimated_num_rows"] for split_size in split_sizes)
                else None,
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

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_config_size_response(dataset=self.dataset, config=self.config))
