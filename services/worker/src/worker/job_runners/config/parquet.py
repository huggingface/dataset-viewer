# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.dtos import CompleteJobResult, ConfigParquetResponse
from worker.job_runners.config.config_job_runner import ConfigJobRunner


def compute_parquet_response(dataset: str, config: str) -> ConfigParquetResponse:
    """
    Get the response of 'config-parquet' for one specific dataset on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        config (`str`):
            A configuration name.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format

    Returns:
        `ConfigParquetResponse`: An object with the parquet_response (list of parquet files).
    """
    logging.info(f"compute 'config-parquet' for {dataset=} {config=}")

    previous_step = "config-parquet-and-info"
    config_parquet_and_info_response = get_previous_step_or_raise(kind=previous_step, dataset=dataset, config=config)
    content = config_parquet_and_info_response["content"]

    try:
        parquet_files = [
            parquet_file for parquet_file in content["parquet_files"] if parquet_file.get("config") == config
        ]
        # sort by filename, which ensures the shards are in order: 00000, 00001, 00002, ...
        parquet_files.sort(key=lambda x: (x["split"], x["filename"]))
        if "features" in content["dataset_info"] and isinstance(content["dataset_info"]["features"], dict):
            features = content["dataset_info"]["features"]
        else:
            # (July 23) we can remove this later and raise an error instead (can be None for backward compatibility)
            features = None
        partial = content["partial"]
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'parquet_files'.", e) from e
    return ConfigParquetResponse(parquet_files=parquet_files, features=features, partial=partial)


class ConfigParquetJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-parquet"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_parquet_response(dataset=self.dataset, config=self.config))
