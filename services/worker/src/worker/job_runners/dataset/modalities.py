# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
)

from worker.dtos import (
    DatasetModalitiesResponse,
    JobResult,
    Modality,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_modalities_response(dataset: str) -> tuple[DatasetModalitiesResponse, float]:
    """
    Get the response of 'dataset-modalities' for one specific dataset on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
            If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `tuple[DatasetModalitiesResponse, float]`: An object with the modalities_response and the progress.
    """
    logging.info(f"compute 'dataset-modalities' for {dataset=}")

    dataset_info_response = get_previous_step_or_raise(kind="dataset-info", dataset=dataset)
    content = dataset_info_response["content"]
    if "dataset_info" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.")

    try:
        modalities: set[Modality] = set()
        # TODO
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return (
        DatasetModalitiesResponse(
            {
                "modalities": sorted(modalities),
                "pending": content["pending"],
                "failed": content["failed"],
                "partial": content["partial"],
            }
        ),
        dataset_info_response["progress"] or 1.0,
    )


class DatasetModalitiesJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-modalities"

    def compute(self) -> JobResult:
        response_content, progress = compute_modalities_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
