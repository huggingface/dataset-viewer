# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from libcommon.constants import (
    PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_INFO_VERSION,
    PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_STREAMING_VERSION,
)
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import CompleteJobResult, JobRunnerInfo, SplitItem, SplitsList


def compute_split_names_from_info_response(dataset: str, config: str) -> SplitsList:
    """
    Get the response of 'config-split-names-from-info' for one specific dataset and config on huggingface.co
    computed from cached response in dataset-info step.

    The 'config-split-names-from-info' response generated by this function does not include stats about the split,
    like the size or number of samples. See dataset-info or dataset-size for that.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `SplitsList`: An object with the list of split names for the dataset and config.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
    """
    logging.info(f"get split names from dataset info for dataset={dataset}, config={config}")
    config_info_best_response = get_previous_step_or_raise(kinds=["config-info"], dataset=dataset, config=config)

    try:
        splits_content = config_info_best_response.response["content"]["dataset_info"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step 'config-info' did not return the expected content.") from e

    split_name_items: List[SplitItem] = [
        {"dataset": dataset, "config": config, "split": str(split)} for split in splits_content
    ]

    return SplitsList(splits=split_name_items)


class ConfigSplitNamesFromInfoJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-split-names-from-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_INFO_VERSION

    @staticmethod
    def get_parallel_job_runner() -> JobRunnerInfo:
        return JobRunnerInfo(
            job_runner_version=PROCESSING_STEP_CONFIG_SPLIT_NAMES_FROM_STREAMING_VERSION,
            job_type="config-split-names-from-streaming",
        )

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_split_names_from_info_response(dataset=self.dataset, config=self.config))
