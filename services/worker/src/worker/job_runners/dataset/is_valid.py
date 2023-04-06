# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Literal, Mapping, Tuple, TypedDict

from libcommon.constants import PROCESSING_STEP_DATASET_IS_VALID_VERSION
from libcommon.simple_cache import SplitFullName, get_validity_by_kind

from worker.job_runner import JobResult, JobRunner

IsValidJobRunnerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
]


class DatasetIsValidResponse(TypedDict):
    valid: bool


SPLIT_KINDS = [
    "/splits",
    "/split-names-from-streaming",
    "/split-names-from-dataset-info",
]
FIRST_ROWS_KINDS = ["split-first-rows-from-streaming", "split-first-rows-from-parquet"]


def compute_is_valid_response(dataset: str) -> Tuple[DatasetIsValidResponse, float]:
    """
    Get the response of /is-valid for one specific dataset on huggingface.co.

    A dataset is valid if:
    - /splits is valid
    - /first-rows is valid for at least one of the splits

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `DatasetIsValidResponse`: An object with the is_valid_response.
    """
    logging.info(f"get is-valid response for dataset={dataset}")

    validity_by_kind = get_validity_by_kind(dataset=dataset, kinds=SPLIT_KINDS + FIRST_ROWS_KINDS)
    is_valid = any(validity_by_kind[kind] for kind in SPLIT_KINDS if kind in validity_by_kind) and any(
        validity_by_kind[kind] for kind in FIRST_ROWS_KINDS if kind in validity_by_kind
    )

    return (DatasetIsValidResponse({"valid": is_valid}), 1.0)


class DatasetIsValidJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-is-valid"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_IS_VALID_VERSION

    def compute(self) -> JobResult:
        if self.dataset is None:
            raise ValueError("dataset is required")
        response_content, progress = compute_is_valid_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return set()
        # ^ it does not make sense to depend on this step. Not sure what should be returned here.
