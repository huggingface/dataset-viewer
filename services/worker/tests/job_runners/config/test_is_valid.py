# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable, List

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.config.is_valid import ConfigIsValidJobRunner

from ..utils import UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigIsValidJobRunner]

DATASET = "dataset"
CONFIG = "config"
SPLIT_1 = "split1"
SPLIT_2 = "split2"

UPSTREAM_RESPONSE_SPLIT_NAMES: UpstreamResponse = UpstreamResponse(
    kind="config-split-names-from-streaming",
    dataset=DATASET,
    config=CONFIG,
    http_status=HTTPStatus.OK,
    content={
        "splits": [
            {"dataset": DATASET, "config": CONFIG, "split": SPLIT_1},
            {"dataset": DATASET, "config": CONFIG, "split": SPLIT_2},
        ]
    },
)
UPSTREAM_RESPONSE_SPLIT_1_OK: UpstreamResponse = UpstreamResponse(
    kind="split-is-valid",
    dataset=DATASET,
    config=CONFIG,
    split=SPLIT_1,
    http_status=HTTPStatus.OK,
    content={"viewer": True, "preview": True, "search": True},
)
UPSTREAM_RESPONSE_SPLIT_1_OK_VIEWER: UpstreamResponse = UpstreamResponse(
    kind="split-is-valid",
    dataset=DATASET,
    config=CONFIG,
    split=SPLIT_1,
    http_status=HTTPStatus.OK,
    content={"viewer": True, "preview": False, "search": False},
)
UPSTREAM_RESPONSE_SPLIT_2_OK_SEARCH: UpstreamResponse = UpstreamResponse(
    kind="split-is-valid",
    dataset=DATASET,
    config=CONFIG,
    split=SPLIT_2,
    http_status=HTTPStatus.OK,
    content={"viewer": False, "preview": False, "search": True},
)
UPSTREAM_RESPONSE_SPLIT_2_OK: UpstreamResponse = UpstreamResponse(
    kind="split-is-valid",
    dataset=DATASET,
    config=CONFIG,
    split=SPLIT_2,
    http_status=HTTPStatus.OK,
    content={"viewer": True, "preview": True, "search": True},
)
UPSTREAM_RESPONSE_SPLIT_1_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-is-valid",
    dataset=DATASET,
    config=CONFIG,
    split=SPLIT_1,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_2_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-is-valid",
    dataset=DATASET,
    config=CONFIG,
    split=SPLIT_2,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
EXPECTED_COMPLETED_ALL_FALSE = (
    {"viewer": False, "preview": False, "search": False},
    1.0,
)
EXPECTED_ALL_MIXED = (
    {"viewer": True, "preview": False, "search": True},
    1.0,
)
EXPECTED_COMPLETED_ALL_TRUE = (
    {"viewer": True, "preview": True, "search": True},
    1.0,
)
EXPECTED_PENDING_ALL_TRUE = (
    {"viewer": True, "preview": True, "search": True},
    0.5,
)
EXPECTED_PENDING_ALL_FALSE = (
    {"viewer": False, "preview": False, "search": False},
    0.0,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigIsValidJobRunner:
        processing_step_name = ConfigIsValidJobRunner.get_job_type()
        processing_graph = ProcessingGraph(app_config.processing_graph.specification)
        return ConfigIsValidJobRunner(
            job_info={
                "type": ConfigIsValidJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": config,
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
                UPSTREAM_RESPONSE_SPLIT_1_OK,
                UPSTREAM_RESPONSE_SPLIT_2_OK,
            ],
            EXPECTED_COMPLETED_ALL_TRUE,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_1_OK,
            ],
            EXPECTED_PENDING_ALL_TRUE,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_1_ERROR,
                UPSTREAM_RESPONSE_SPLIT_2_ERROR,
            ],
            EXPECTED_COMPLETED_ALL_FALSE,
        ),
        ([UPSTREAM_RESPONSE_SPLIT_1_OK_VIEWER, UPSTREAM_RESPONSE_SPLIT_2_OK_SEARCH], EXPECTED_ALL_MIXED),
        (
            [],
            EXPECTED_PENDING_ALL_FALSE,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: List[UpstreamResponse],
    expected: Any,
) -> None:
    dataset, config = DATASET, CONFIG
    upsert_response(**UPSTREAM_RESPONSE_SPLIT_NAMES)
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, config, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]
