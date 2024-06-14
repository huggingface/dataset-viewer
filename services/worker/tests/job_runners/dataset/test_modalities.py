# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from datasets import Features, Image, Value
from libcommon.dtos import Priority
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.dtos import DatasetModalitiesResponse
from worker.job_runners.dataset.modalities import DatasetModalitiesJobRunner
from worker.resources import LibrariesResource

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetModalitiesJobRunner]

TEXT_DATASET = "text-dataset"
IMAGE_TEXT_DATASET = "image-text-dataset"
ERROR_DATASET = "error-dataset"

text_features = Features({"conversations": [{"from": Value("string"), "value": Value("string")}]})
image_text_features = Features({"image": Image(), "caption": Value("string")})

UPSTREAM_RESPONSE_INFO_TEXT: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": text_features.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_IMAGE_TEXT: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=IMAGE_TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": image_text_features.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
UPSTREAM_RESPONSE_INFO_MALFORMED: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    # The content is missing the "dataset_info" key
    content={"bad": "content"},
    progress=0.0,
)

EXPECTED_TEXT: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["text"]},
    1.0,
)
EXPECTED_IMAGE_TEXT: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["image", "text"]},
    1.0,
)
EXPECTED_EMPTY: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": []},
    1.0,
)


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetModalitiesJobRunner:
        return DatasetModalitiesJobRunner(
            job_info={
                "type": DatasetModalitiesJobRunner.get_job_type(),
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
    "dataset,upstream_responses,expected",
    [
        (
            TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_TEXT,
            ],
            EXPECTED_TEXT,
        ),
        (
            IMAGE_TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_IMAGE_TEXT,
            ],
            EXPECTED_IMAGE_TEXT,
        ),
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_ERROR,
            ],
            EXPECTED_EMPTY,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


@pytest.mark.parametrize(
    "dataset,upstream_responses,expectation",
    [
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_MALFORMED,
            ],
            pytest.raises(PreviousStepFormatError),
        )
    ],
)
def test_compute_error(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expectation: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    with expectation:
        job_runner.compute()
