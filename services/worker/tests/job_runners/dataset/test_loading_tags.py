# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response

from worker.config import AppConfig
from worker.job_runners.dataset.loading_tags import DatasetLoadingTagsJobRunner

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetLoadingTagsJobRunner]

DATASET = "dataset"

UPSTREAM_RESPONSE_INFO_PARQUET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"builder_name": "parquet"}}},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_WEBDATASET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"builder_name": "webdataset"}}},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFD_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
EXPECTED_PARQUET = (
    {"tags": ["croissant"]},
    1.0,
)
EXPECTED_WEBDATASET = (
    {"tags": ["croissant", "webdataset"]},
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
    ) -> DatasetLoadingTagsJobRunner:
        return DatasetLoadingTagsJobRunner(
            job_info={
                "type": DatasetLoadingTagsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 20,
            },
            app_config=app_config,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "upstream_responses,expected",
    [
        (
            [
                UPSTREAM_RESPONSE_INFO_PARQUET,
            ],
            EXPECTED_PARQUET,
        ),
        (
            [
                UPSTREAM_RESPONSE_INFO_WEBDATASET,
            ],
            EXPECTED_WEBDATASET,
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
                UPSTREAM_RESPONSE_INFD_ERROR,
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
