# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.dtos import DatasetHubCacheResponse, JobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_hub_cache_response(dataset: str) -> tuple[DatasetHubCacheResponse, float]:
    """
    Get the response of `dataset-hub-cache` for one specific dataset on huggingface.co.
    Its purpose is specific to the Hub, and we won't ensure backward compatibility for this step.
    It provides information about:
    - the capabilities of the dataset: preview and viewer
    - the number of rows and if the dataset is partial

    Args:
        dataset (:obj:`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        PreviousStepFormatError:
            If the content of the previous step has not the expected format

    Returns:
        :obj:`tuple[DatasetHubCacheResponse, float]`: 
            A tuple containing the response and the progress.
    """
    logging.info(f"get 'dataset-hub-cache' response for {dataset=}")

    is_valid_response = get_previous_step_or_raise(kinds=["dataset-is-valid"], dataset=dataset)
    content = is_valid_response.response["content"]
    if (
        "preview" not in content
        or not isinstance(content["preview"], bool)
        or "viewer" not in content
        or not isinstance(content["viewer"], bool)
    ):
        raise PreviousStepFormatError(
            "Previous step 'dataset-is-valid' did not return the expected content: 'preview', 'viewer' or 'progress'."
        )
    preview = content["preview"]
    viewer = content["viewer"]
    is_valid_progress = is_valid_response.response["progress"]

    size_response = get_previous_step_or_raise(kinds=["dataset-size"], dataset=dataset)
    content = size_response.response["content"]
    if (
        "partial" not in content
        or not isinstance(content["partial"], bool)
        or "size" not in content
        or "dataset" not in content["size"]
        or "num_rows" not in content["size"]["dataset"]
        or not isinstance(content["size"]["dataset"]["num_rows"], int)
    ):
        raise PreviousStepFormatError(
            "Previous step 'dataset-size' did not return the expected content: 'partial' or 'size.dataset.num_rows'."
        )

    partial = content["partial"]
    num_rows = content["size"]["dataset"]["num_rows"]
    size_progress = size_response.response["progress"]

    progress = min((p for p in [is_valid_progress, size_progress] if p is not None), default=0.0)

    return (
        DatasetHubCacheResponse(
            preview=preview,
            viewer=viewer,
            partial=partial,
            num_rows=num_rows,
        ),
        progress,
    )


class DatasetHubCacheJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-hub-cache"

    def compute(self) -> JobResult:
        response_content, progress = compute_hub_cache_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
