# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_CONFIG_PARQUET_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import SplitHubFile

from worker.dtos import CompleteJobResult, ConfigParquetResponse
from worker.job_runners.config.config_job_runner import ConfigJobRunner


def compute_parquet_response(dataset: str, config: str) -> ConfigParquetResponse:
    """
    Get the response of /parquet for one specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `ConfigParquetResponse`: An object with the parquet_response (list of parquet files).
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
    """
    logging.info(f"get parquet files for dataset={dataset}, config={config}")

    previous_step = "config-parquet-and-info"
    config_parquet_and_info_best_response = get_previous_step_or_raise(
        kinds=[previous_step], dataset=dataset, config=config
    )
    content = config_parquet_and_info_best_response.response["content"]
    if "parquet_files" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'parquet_files'.")

    parquet_files = [parquet_file for parquet_file in content["parquet_files"] if parquet_file.get("config") == config]
    return ConfigParquetResponse(parquet_files=parquet_files)


class ConfigParquetJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_PARQUET_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_parquet_response(dataset=self.dataset, config=self.config))
