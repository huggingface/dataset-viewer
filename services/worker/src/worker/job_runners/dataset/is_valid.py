# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Tuple

from libcommon.constants import PROCESSING_STEP_DATASET_IS_VALID_VERSION
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.simple_cache import is_valid_for_kinds
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.dtos import DatasetIsValidResponse, JobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_is_valid_response(dataset: str, processing_graph: ProcessingGraph) -> Tuple[DatasetIsValidResponse, float]:
    """
    Get the response of /is-valid for one specific dataset on huggingface.co.


    A dataset is valid if at least one response of any of the artifacts for any of the
    steps (for viewer and preview) is valid.
    The deprecated `valid` field is an "or" of the `preview` and `viewer` fields.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        processing_graph (`ProcessingGraph`):
            The processing graph. In particular, it must provide the list of
            processing steps that enable the viewer and the preview.
    Returns:
        `DatasetIsValidResponse`: The response (viewer, preview, valid).
    """
    logging.info(f"get is-valid response for dataset={dataset}")

    viewer = is_valid_for_kinds(
        dataset=dataset, kinds=[step.cache_kind for step in processing_graph.get_processing_steps_enables_viewer()]
    )
    preview = is_valid_for_kinds(
        dataset=dataset, kinds=[step.cache_kind for step in processing_graph.get_processing_steps_enables_preview()]
    )

    return (DatasetIsValidResponse({"viewer": viewer, "preview": preview}), 1.0)


class DatasetIsValidJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-is-valid"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_IS_VALID_VERSION

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
        if self.dataset is None:
            raise ValueError("dataset is required")
        response_content, progress = compute_is_valid_response(
            dataset=self.dataset, processing_graph=self.processing_graph
        )
        return JobResult(response_content, progress=progress)
