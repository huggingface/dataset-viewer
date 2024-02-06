# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from itertools import islice

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
)

from worker.dtos import (
    CompleteJobResult,
    DatasetLoadingTag,
    DatasetLoadingTagsResponse,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner

MAX_CONFIGS = 100


def compute_loading_tags_response(dataset: str) -> DatasetLoadingTagsResponse:
    """
    Get the response of 'dataset-loading-tags' for one specific dataset on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `DatasetLoadingTagsResponse`: The dataset-loading-tags response (list of tags).
    """
    logging.info(f"get 'dataset-loading-tags' for {dataset=}")

    dataset_info_best_response = get_previous_step_or_raise(kinds=["dataset-info"], dataset=dataset)
    http_status = dataset_info_best_response.response["http_status"]
    tags: list[DatasetLoadingTag] = []
    if http_status == HTTPStatus.OK:
        try:
            content = dataset_info_best_response.response["content"]
            infos = list(islice(content["dataset_info"].values(), MAX_CONFIGS))
            if infos:
                tags.append("croissant")
                if infos[0]["builder_name"] == "webdataset":
                    tags.append("webdataset")
        except KeyError as e:
            raise PreviousStepFormatError(
                "Previous step 'dataset-info' did not return the expected content.", e
            ) from e
    return DatasetLoadingTagsResponse(tags=tags)


class DatasetLoadingTagsJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-loading-tags"

    def compute(self) -> CompleteJobResult:
        response_content = compute_loading_tags_response(dataset=self.dataset)
        return CompleteJobResult(response_content)
