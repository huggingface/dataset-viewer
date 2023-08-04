# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_SPLIT_IS_VALID_VERSION
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.simple_cache import has_any_successful_response
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.dtos import CompleteJobResult, IsValidResponse, JobResult
from worker.job_runners.split.split_job_runner import SplitJobRunner


def compute_is_valid_response(
    dataset: str, config: str, split: str, processing_graph: ProcessingGraph
) -> IsValidResponse:
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
        processing_graph (`ProcessingGraph`):
            The processing graph. In particular, it must provide the list of
            processing steps that enable the viewer and the preview.
    Returns:
        `IsValidResponse`: The response (viewer, preview, search).
    """
    logging.info(f"get is-valid response for dataset={dataset}")

    viewer = has_any_successful_response(
        dataset=dataset,
        config=config,
        split=None,
        kinds=[step.cache_kind for step in processing_graph.get_processing_steps_enables_viewer()],
    )
    preview = has_any_successful_response(
        dataset=dataset,
        config=config,
        split=split,
        kinds=[step.cache_kind for step in processing_graph.get_processing_steps_enables_preview()],
    )
    search = has_any_successful_response(
        dataset=dataset,
        config=config,
        split=split,
        kinds=[step.cache_kind for step in processing_graph.get_processing_steps_enables_search()],
    )
    return IsValidResponse(viewer=viewer, preview=preview, search=search)


class SplitIsValidJobRunner(SplitJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "split-is-valid"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_IS_VALID_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        processing_graph: ProcessingGraph,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
        self.processing_graph = processing_graph

    def compute(self) -> JobResult:
        return CompleteJobResult(
            compute_is_valid_response(
                dataset=self.dataset, config=self.config, split=self.split, processing_graph=self.processing_graph
            )
        )
