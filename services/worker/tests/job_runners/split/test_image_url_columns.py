# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable, Mapping
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.config import ProcessingGraphConfig
from libcommon.constants import (
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
    PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.dtos import ImageUrlColumnsResponse
from worker.job_runners.split.image_url_columns import SplitImageUrlColumnsJobRunner

from ...fixtures.hub import get_default_config_split

GetJobRunner = Callable[[str, str, str, AppConfig], SplitImageUrlColumnsJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitImageUrlColumnsJobRunner:
        processing_step_name = SplitImageUrlColumnsJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            ProcessingGraphConfig(
                {
                    "dataset-level": {"input_type": "dataset"},
                    "config-level": {"input_type": "dataset", "triggered_by": "dataset-level"},
                    processing_step_name: {
                        "input_type": "dataset",
                        "job_runner_version": SplitImageUrlColumnsJobRunner.get_job_runner_version(),
                        "triggered_by": "config-level",
                    },
                }
            )
        )

        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names-from-streaming",
            dataset=dataset,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitImageUrlColumnsJobRunner(
            job_info={
                "type": SplitImageUrlColumnsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
        )

    return _get_job_runner


FIRST_ROWS_WITHOUT_STR_COLUMNS = {
    "features": [
        {
            "feature_idx": 0,
            "name": "col1",
            "type": {
                "dtype": "int64",
                "_type": "Value",
            },
        },
        {
            "feature_idx": 1,
            "name": "col2",
            "type": {
                "dtype": "float",
                "_type": "Value",
            },
        },
    ],
    "rows": [],
}


FIRST_ROWS_WITHOUT_IMAGE_URL_COLUMNS = {
    "features": [
        {
            "feature_idx": 0,
            "name": "col1",
            "type": {
                "dtype": "string",
                "_type": "Value",
            },
        },
    ],
    "rows": [
        {"row_idx": 0, "row": {"col": "http://testurl.test/test_document.txt"}, "truncated_cells": []},
        {"row_idx": 1, "row": {"col": "http://testurl.test/test"}, "truncated_cells": []},
    ],
}


FIRST_ROWS_WITH_IMAGE_URL_COLUMNS = {
    "features": [
        {
            "feature_idx": 0,
            "name": "col",
            "type": {
                "dtype": "string",
                "_type": "Value",
            },
        },
        {
            "feature_idx": 1,
            "name": "col1",
            "type": {
                "dtype": "string",
                "_type": "Value",
            },
        },
    ],
    "rows": [
        {"row_idx": 0, "row": {"col": "http://testurl.test/test_image.jpg", "col1": ""}, "truncated_cells": []},
        {"row_idx": 1, "row": {"col": "http://testurl.test/test_image2.jpg"}, "col1": "text", "truncated_cells": []},
        {"row_idx": 2, "row": {"col": "other", "col1": "text"}, "truncated_cells": []},
        {"row_idx": 1, "row": {"col": "http://testurl.test/test_image3.png", "col1": "text"}, "truncated_cells": []},
    ],
}


FIRST_ROWS_WITH_IMAGE_URL_COLUMNS_NO_ROWS = {
    "features": [
        {
            "feature_idx": 0,
            "name": "col",
            "type": {
                "dtype": "string",
                "_type": "Value",
            },
        },
    ],
    "rows": [],
}


DEFAULT_EMPTY_CONTENT: ImageUrlColumnsResponse = {"columns": []}


@pytest.mark.parametrize(
    "dataset,upstream_content,expected_content",
    [
        (
            "no_str_columns",
            FIRST_ROWS_WITHOUT_STR_COLUMNS,
            DEFAULT_EMPTY_CONTENT,
        ),
        (
            "no_image_url_columns",
            FIRST_ROWS_WITHOUT_IMAGE_URL_COLUMNS,
            DEFAULT_EMPTY_CONTENT,
        ),
        (
            "image_url_columns",
            FIRST_ROWS_WITH_IMAGE_URL_COLUMNS,
            {"columns": ["col"]},
        ),
        (
            "image_url_columns_no_rows",
            FIRST_ROWS_WITH_IMAGE_URL_COLUMNS_NO_ROWS,
            DEFAULT_EMPTY_CONTENT,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_content: Mapping[str, Any],
    expected_content: Mapping[str, Any],
) -> None:
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config,
    )
    upsert_response(
        kind="split-first-rows-from-streaming",
        dataset=dataset,
        config=config,
        split=split,
        content=upstream_content,
        dataset_git_revision="dataset_git_revision",
        job_runner_version=PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    response = job_runner.compute()
    assert response
    assert response.content == expected_content


@pytest.mark.parametrize(
    "dataset,upstream_content,upstream_status,exception_name",
    [
        ("doesnotexist", {}, HTTPStatus.OK, "CachedArtifactNotFoundError"),
        ("wrong_format", {}, HTTPStatus.OK, "PreviousStepFormatError"),
        (
            "upstream_failed",
            {},
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "CachedArtifactError",
        ),
    ],
)
def test_compute_failed(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_content: Mapping[str, Any],
    upstream_status: HTTPStatus,
    exception_name: str,
) -> None:
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config,
    )
    if dataset != "doesnotexist":
        upsert_response(
            kind="split-first-rows-from-streaming",
            dataset=dataset,
            config=config,
            split=split,
            content=upstream_content,
            dataset_git_revision="dataset_git_revision",
            job_runner_version=PROCESSING_STEP_SPLIT_IMAGE_URL_COLUMNS_VERSION,
            progress=1.0,
            http_status=upstream_status,
        )
    with pytest.raises(Exception) as exc_info:
        job_runner.compute()
    assert exc_info.typename == exception_name
