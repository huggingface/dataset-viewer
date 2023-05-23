# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import List, Tuple

from libcommon.constants import PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_best_response

from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import (
    ConfigItem,
    DatasetSplitNamesResponse,
    FailedConfigItem,
    JobResult,
    SplitItem,
    get_previous_step_or_raise,
)


def compute_dataset_split_names_response(dataset: str) -> Tuple[DatasetSplitNamesResponse, float]:
    """
    Get the response of /splits for one specific dataset on huggingface.co
    computed from responses cached in 'config-split-names-from-info' or 'config-split-names-from-streaming' steps.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
    Returns:
        `DatasetSplitNamesResponse`: An object with a list of split names for the dataset [splits],
         a list of pending configs to be processed [pending] and the list of errors [failed] by config.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    """
    logging.info(f"get dataset split names for dataset={dataset}")

    # Get the config names from the previous steps
    config_names_best_response = get_previous_step_or_raise(kinds=["/config-names"], dataset=dataset)
    content = config_names_best_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("'/config-names' did not return the expected content: 'config_names'.")
    config_names = [config_name_item["config"] for config_name_item in content["config_names"]]
    if any(not isinstance(config_name, str) for config_name in config_names):
        raise PreviousStepFormatError("Previous step '/config-names' did not return a list of config names.")

    split_names_cache_kinds = ["config-split-names-from-info", "config-split-names-from-streaming"]
    try:
        splits: List[SplitItem] = []
        pending: List[ConfigItem] = []
        failed: List[FailedConfigItem] = []
        total = 0
        for config in config_names:
            total += 1
            best_response = get_best_response(split_names_cache_kinds, dataset=dataset, config=config)
            if best_response.response["error_code"] == "CachedResponseNotFound":
                logging.debug(
                    "No response (successful or erroneous) found in cache for the previous steps"
                    f" '{split_names_cache_kinds}' for this dataset."
                )
                pending.append(ConfigItem({"dataset": dataset, "config": config}))
                continue
            if best_response.response["http_status"] != HTTPStatus.OK:
                logging.debug(f"No successful response found in the previous steps {split_names_cache_kinds}.")
                failed.append(
                    FailedConfigItem(
                        {
                            "dataset": dataset,
                            "config": config,
                            "error": best_response.response["content"],
                        }
                    )
                )
                continue
            splits.extend(
                [
                    SplitItem({"dataset": dataset, "config": config, "split": split_content["split"]})
                    for split_content in best_response.response["content"]["splits"]
                ]
            )
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - len(pending)) / total if total else 1.0

    return (
        DatasetSplitNamesResponse(
            {
                "splits": splits,
                "pending": pending,
                "failed": failed,
            }
        ),
        progress,
    )


class DatasetSplitNamesJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-split-names"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_dataset_split_names_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
