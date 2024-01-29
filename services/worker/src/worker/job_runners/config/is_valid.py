# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CacheEntryDoesNotExistError,
    get_response,
)

from worker.dtos import IsValidResponse, JobResult
from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import get_split_names


def compute_is_valid_response(dataset: str, config: str) -> tuple[IsValidResponse, float]:
    """
    Get the response of 'config-is-valid' for one specific dataset config on huggingface.co.

    A dataset config is valid if any of the artifacts for any of the
    steps is valid.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `tuple[IsValidResponse, float]`: The response (viewer, preview, search, filter) and the progress.
    """
    logging.info(f"get 'config-is-valid' response for {dataset=} {config=}")

    preview = False
    viewer = False
    search = False
    filter = False
    try:
        total = 0
        pending = 0
        for split in get_split_names(dataset=dataset, config=config):
            total += 1
            try:
                response = get_response(kind="split-is-valid", dataset=dataset, config=config, split=split)
            except CacheEntryDoesNotExistError:
                logging.debug("No response found in previous step for this dataset: 'split-is-valid'.")
                pending += 1
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                continue
            split_is_valid_content = response["content"]
            preview = preview or split_is_valid_content["preview"]
            viewer = viewer or split_is_valid_content["viewer"]
            search = search or split_is_valid_content["search"]
            filter = filter or split_is_valid_content["filter"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    progress = (total - pending) / total if total else 1.0
    return (
        IsValidResponse(preview=preview, viewer=viewer, search=search, filter=filter),
        progress,
    )


class ConfigIsValidJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-is-valid"

    def compute(self) -> JobResult:
        response_content, progress = compute_is_valid_response(dataset=self.dataset, config=self.config)
        return JobResult(response_content, progress=progress)
