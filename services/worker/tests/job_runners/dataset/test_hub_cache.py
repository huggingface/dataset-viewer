# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.dataset.hub_cache import DatasetHubCacheJobRunner

from ..utils import UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetHubCacheJobRunner]

DATASET = "dataset"

UPSTREAM_RESPONSE_IS_VALID_OK: UpstreamResponse = UpstreamResponse(
    kind="dataset-is-valid",
    dataset=DATASET,
    http_status=HTTPStatus.OK,
    content={"preview": True, "viewer": False, "search": True},
    progress=0.5,
)
UPSTREAM_RESPONSE_IS_VALID_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-is-valid",
    dataset=DATASET,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
UPSTREAM_RESPONSE_SIZE_OK: UpstreamResponse = UpstreamResponse(
    kind="dataset-size",
    dataset=DATASET,
    http_status=HTTPStatus.OK,
    content={"size": {"dataset": {"num_rows": 1000}}, "partial": False},
    progress=0.2,
)
UPSTREAM_RESPONSE_SIZE_NO_PROGRESS: UpstreamResponse = UpstreamResponse(
    kind="dataset-size",
    dataset=DATASET,
    http_status=HTTPStatus.OK,
    content={"size": {"dataset": {"num_rows": 1000}}, "partial": True},
    progress=None,
)
EXPECTED_OK = (
    {"viewer": False, "preview": True, "partial": False, "num_rows": 1000},
    0.2,
)
EXPECTED_NO_PROGRESS = (
    {"viewer": False, "preview": True, "partial": True, "num_rows": 1000},
    0.5,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetHubCacheJobRunner:
        processing_step_name = DatasetHubCacheJobRunner.get_job_type()
        processing_graph = ProcessingGraph(app_config.processing_graph)
        return DatasetHubCacheJobRunner(
            job_info={
                "type": DatasetHubCacheJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                    "revision": "revision",
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 20,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "upstream_responses,expected",
    [
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_OK,
            ],
            EXPECTED_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_NO_PROGRESS,
            ],
            EXPECTED_NO_PROGRESS,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    dataset = DATASET
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


@pytest.mark.parametrize(
    "upstream_responses,expectation",
    [
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_ERROR,
                UPSTREAM_RESPONSE_SIZE_OK,
            ],
            pytest.raises(CachedArtifactError),
        )
    ],
)
def test_compute_error(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: list[UpstreamResponse],
    expectation: Any,
) -> None:
    dataset = DATASET
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    with expectation:
        job_runner.compute()
