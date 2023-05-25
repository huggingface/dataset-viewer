# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.utils import is_image_url

from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import (
    CompleteJobResult,
    ImageUrlColumnsResponse,
    SplitFirstRowsResponse,
    get_previous_step_or_raise,
)


def compute_image_url_columns(
    dataset: str,
    config: str,
    split: str,
) -> ImageUrlColumnsResponse:
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

    # look for URLs columns using the first rows
    string_type_dict = {"dtype": "string", "_type": "Value"}
    string_columns = [feature["name"] for feature in features if feature["type"] == string_type_dict]
    urls_columns = []
    for string_column in string_columns:
        urls_count = sum(
            1
            for row in first_rows
            if isinstance(row["row"].get(string_column), str) and is_image_url(text=row["row"][string_column])
        )
        if urls_count and urls_count / len(first_rows) > 0.3:
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
