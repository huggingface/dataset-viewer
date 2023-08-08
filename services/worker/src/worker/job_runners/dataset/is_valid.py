# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Tuple

from libcommon.constants import PROCESSING_STEP_DATASET_IS_VALID_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CacheEntryDoesNotExistError,
    get_previous_step_or_raise,
    get_response,
)

from worker.dtos import IsValidResponse, JobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_is_valid_response(dataset: str) -> Tuple[IsValidResponse, float]:
    """
    Get the response of /is-valid for one specific dataset on huggingface.co.


    A dataset is valid if at least one response of any of the artifacts for any of the
    steps (for viewer and preview) is valid.
    The deprecated `valid` field is an "or" of the `preview` and `viewer` fields.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `Tuple[IsValidResponse, float]`: The response (viewer, preview, search) and the progress.
    """
    logging.info(f"get is-valid response for {dataset=}")

    config_names_response = get_previous_step_or_raise(kinds=["dataset-config-names"], dataset=dataset)
    content = config_names_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    preview = False
    viewer = False
    search = False
    try:
        total = 0
        pending = 0
        for config_item in content["config_names"]:
            config = config_item["config"]
            total += 1
            try:
                response = get_response(kind="config-is-valid", dataset=dataset, config=config)
            except CacheEntryDoesNotExistError:
                logging.debug("No response found in previous step for this dataset: 'config-is-valid'.")
                pending += 1
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                continue
            config_is_valid_content = response["content"]
            preview = preview or config_is_valid_content["preview"]
            viewer = viewer or config_is_valid_content["viewer"]
            search = search or config_is_valid_content["search"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - pending) / total if total else 1.0

    return (
        IsValidResponse(
            preview=preview,
            viewer=viewer,
            search=search,
        ),
        progress,
    )


class DatasetIsValidJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-is-valid"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_IS_VALID_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_is_valid_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
