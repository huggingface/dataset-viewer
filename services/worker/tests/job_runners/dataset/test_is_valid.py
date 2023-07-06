# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable, List

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.dataset.is_valid import DatasetIsValidJobRunner

from ..utils import UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetIsValidJobRunner]

DATASET = "dataset"

UPSTREAM_RESPONSE_CONFIG_SIZE: UpstreamResponse = UpstreamResponse(
    kind="config-size", dataset=DATASET, config="config", http_status=HTTPStatus.OK, content={}
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-parquet",
    dataset=DATASET,
    config="config",
    split="split",
    http_status=HTTPStatus.OK,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-streaming",
    dataset=DATASET,
    config="config",
    split="split",
    http_status=HTTPStatus.OK,
    content={},
)
UPSTREAM_RESPONSE_CONFIG_SIZE_ERROR: UpstreamResponse = UpstreamResponse(
    kind="config-size", dataset=DATASET, config="config", http_status=HTTPStatus.INTERNAL_SERVER_ERROR, content={}
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-parquet",
    dataset=DATASET,
    config="config",
    split="split",
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-streaming",
    dataset=DATASET,
    config="config",
    split="split",
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
EXPECTED_ERROR = (
    {"viewer": False, "preview": False},
    1.0,
)
EXPECTED_VIEWER_OK = (
    {"viewer": True, "preview": False},
    1.0,
)
EXPECTED_PREVIEW_OK = (
    {"viewer": False, "preview": True},
    1.0,
)
EXPECTED_BOTH_OK = (
    {"viewer": True, "preview": True},
    1.0,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetIsValidJobRunner:
        processing_step_name = DatasetIsValidJobRunner.get_job_type()
        processing_graph = ProcessingGraph(app_config.processing_graph.specification)
        return DatasetIsValidJobRunner(
            job_info={
                "type": DatasetIsValidJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": None,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            processing_graph=processing_graph,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "upstream_responses,expected",
    [
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING,
            ],
            EXPECTED_BOTH_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING,
            ],
            EXPECTED_PREVIEW_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING,
            ],
            EXPECTED_PREVIEW_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING,
            ],
            EXPECTED_BOTH_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING_ERROR,
            ],
            EXPECTED_VIEWER_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE,
            ],
            EXPECTED_VIEWER_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING_ERROR,
            ],
            EXPECTED_ERROR,
        ),
        (
            [],
            EXPECTED_ERROR,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: List[UpstreamResponse],
    expected: Any,
) -> None:
    dataset = DATASET
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == {"viewer": False, "preview": False}
