# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import CONFIG_HAS_VIEWER_KINDS, SPLIT_HAS_PREVIEW_KINDS, SPLIT_HAS_SEARCH_KINDS
from libcommon.simple_cache import (
    get_previous_step_or_raise,
    has_any_successful_response,
)
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.dtos import CompleteJobResult, IsValidResponse, JobResult
from worker.job_runners.split.split_job_runner import SplitJobRunner


def compute_is_valid_response(dataset: str, config: str, split: str) -> IsValidResponse:
    """
    Get the response of /is-valid for one specific dataset split on huggingface.co.


    A dataset split is valid if any of the artifacts for any of the
    steps is valid.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
    Returns:
        `IsValidResponse`: The response (viewer, preview, search, filter).
    """
    logging.info(f"get is-valid response for dataset={dataset}")

    viewer = has_any_successful_response(
        dataset=dataset,
        config=config,
        split=None,
        kinds=CONFIG_HAS_VIEWER_KINDS,
    )
    preview = has_any_successful_response(
        dataset=dataset,
        config=config,
        split=split,
        kinds=SPLIT_HAS_PREVIEW_KINDS,
    )

    try:
        duckdb_response = get_previous_step_or_raise(
            kinds=SPLIT_HAS_SEARCH_KINDS,
            dataset=dataset,
            config=config,
            split=split,
        )
        search_content = duckdb_response.response["content"]
        filter = True
        search = search_content["has_fts"]
    except Exception:
        filter = False
        search = False

    return IsValidResponse(viewer=viewer, preview=preview, search=search, filter=filter)


class SplitIsValidJobRunner(SplitJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "split-is-valid"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
        )

    def compute(self) -> JobResult:
        return CompleteJobResult(compute_is_valid_response(dataset=self.dataset, config=self.config, split=self.split))
