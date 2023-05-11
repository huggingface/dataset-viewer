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


GetJobRunner = Callable[[str, AppConfig, bool], DatasetIsValidJobRunner]


UPSTREAM_RESPONSE_SPLITS: UpstreamResponse = UpstreamResponse(
    kind="/splits", dataset="dataset_ok", config=None, http_status=HTTPStatus.OK, content={}
)
UPSTREAM_RESPONSE_SPLIT_NAMES_FROM_STREAMING: UpstreamResponse = UpstreamResponse(
    kind="config-split-names-from-streaming", dataset="dataset_ok", config=None, http_status=HTTPStatus.OK, content={}
)
UPSTREAM_RESPONSE_SPLIT_NAMES_FROM_DATASET_INFO: UpstreamResponse = UpstreamResponse(
    kind="/split-names-from-dataset-info", dataset="dataset_ok", config=None, http_status=HTTPStatus.OK, content={}
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-streaming",
    dataset="dataset_ok",
    config="config",
    http_status=HTTPStatus.OK,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-parquet", dataset="dataset_ok", config="config", http_status=HTTPStatus.OK, content={}
)
UPSTREAM_RESPONSE_SPLITS_ERROR: UpstreamResponse = UpstreamResponse(
    kind="/splits", dataset="dataset_ok", config=None, http_status=HTTPStatus.INTERNAL_SERVER_ERROR, content={}
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows-from-streaming",
    dataset="dataset_ok",
    config="config",
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
EXPECTED_OK = (
    {"valid": True},
    1.0,
)
EXPECTED_ERROR = (
    {"valid": False},
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
        force: bool = False,
    ) -> DatasetIsValidJobRunner:
        processing_step_name = DatasetIsValidJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": DatasetIsValidJobRunner.get_job_runner_version(),
                }
            }
        )
        return DatasetIsValidJobRunner(
            job_info={
                "type": DatasetIsValidJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                },
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected_error_code,expected,should_raise",
    [
        (
            "dataset_ok",
            [
                UPSTREAM_RESPONSE_SPLITS,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING,
            ],
            None,
            EXPECTED_OK,
            False,
        ),
        (
            "dataset_ok",
            [
                UPSTREAM_RESPONSE_SPLITS,
                UPSTREAM_RESPONSE_SPLIT_NAMES_FROM_STREAMING,
                UPSTREAM_RESPONSE_SPLIT_NAMES_FROM_DATASET_INFO,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET,
            ],
            None,
            EXPECTED_OK,
            False,
        ),
        ("dataset_ok", [], None, EXPECTED_ERROR, False),
        ("dataset_ok", [UPSTREAM_RESPONSE_SPLITS], None, EXPECTED_ERROR, False),
        ("dataset_ok", [UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING], None, EXPECTED_ERROR, False),
        (
            "dataset_ok",
            [UPSTREAM_RESPONSE_SPLITS_ERROR, UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING],
            None,
            EXPECTED_ERROR,
            False,
        ),
        (
            "dataset_ok",
            [UPSTREAM_RESPONSE_SPLITS, UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING_ERROR],
            None,
            EXPECTED_ERROR,
            False,
        ),
        (
            "dataset_ok",
            [
                UPSTREAM_RESPONSE_SPLIT_NAMES_FROM_STREAMING,
                UPSTREAM_RESPONSE_SPLITS_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_PARQUET,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_FROM_STREAMING_ERROR,
            ],
            None,
            EXPECTED_OK,
            False,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: List[UpstreamResponse],
    expected_error_code: str,
    expected: Any,
    should_raise: bool,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config, False)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.type.__name__ == expected_error_code
    else:
        compute_result = job_runner.compute()
        assert compute_result.content == expected[0]
        assert compute_result.progress == expected[1]


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config, False)
    compute_result = job_runner.compute()
    assert compute_result.content == {"valid": False}
