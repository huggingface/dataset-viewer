# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import is_image_url

from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import (
    CompleteJobResult,
    ImageUrlColumnsResponse,
    SplitFirstRowsResponse,
)

STRING_FEATURE_DTYPE = "string"
VALUE_FEATURE_TYPE = "Value"
URL_COLUMN_RATION = 0.3


def compute_image_url_columns(
    dataset: str,
    config: str,
    split: str,
) -> ImageUrlColumnsResponse:
    """
    Get the response of split-image-url-columns cache for a specific split of a dataset from huggingface.co.
    The response is not used directly in the API but it is an input for split-opt-in-out-urls-scan processing step.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
    Returns:
        [`ImageUrlColumnsResponse`]: The list of image url columns.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
    """
    logging.info(f"get image-url-columns for dataset={dataset} config={config} split={split}")

    # get the first rows from previous job
    upstream_response = get_previous_step_or_raise(
        kinds=["split-first-rows-from-streaming", "split-first-rows-from-parquet"],
        dataset=dataset,
        config=config,
        split=split,
    )
    try:
        first_rows_response = upstream_response.response
        upstream_response_content = SplitFirstRowsResponse(
            dataset=dataset,
            config=config,
            split=split,
            features=first_rows_response["content"]["features"],
            rows=first_rows_response["content"]["rows"],
        )

        features = upstream_response_content["features"]
        first_rows = upstream_response_content["rows"]
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    # look for image URLs columns using the first rows
    string_columns = [
        feature["name"]
        for feature in features
        if "dtype" in feature["type"]
        and "_type" in feature["type"]
        and feature["type"]["dtype"] == STRING_FEATURE_DTYPE
        and feature["type"]["_type"] == VALUE_FEATURE_TYPE
    ]

    first_rows_size = len(first_rows)
    if first_rows_size == 0:
        return ImageUrlColumnsResponse(
            columns=[],
        )

    urls_columns = []
    for string_column in string_columns:
        urls_count = sum(
            1
            for row in first_rows
            if isinstance(row["row"].get(string_column), str) and is_image_url(text=row["row"][string_column])
        )
        if urls_count and urls_count / first_rows_size > URL_COLUMN_RATION:
            urls_columns.append(string_column)

    return ImageUrlColumnsResponse(
        columns=urls_columns,
    )


class SplitImageUrlColumnsJobRunner(SplitJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "split-image-url-columns"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_image_url_columns(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
            )
        )
