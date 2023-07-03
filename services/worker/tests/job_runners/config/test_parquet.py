# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable

import pytest
from libcommon.exceptions import PreviousStepFormatError
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from libcommon.utils import Priority, SplitHubFile

from worker.config import AppConfig
from worker.dtos import ConfigParquetAndInfoResponse, ConfigParquetResponse
from worker.job_runners.config.parquet import ConfigParquetJobRunner


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigParquetJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetJobRunner:
        processing_step_name = ConfigParquetJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": ConfigParquetJobRunner.get_job_runner_version(),
                    "triggered_by": "dataset-level",
                },
            }
        )
        return ConfigParquetJobRunner(
            job_info={
                "type": ConfigParquetJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetAndInfoResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url2", filename="filename2", size=0
                    ),
                ],
                dataset_info={"description": "value", "dataset_size": 10},
            ),
            None,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url2", filename="filename2", size=0
                    ),
                ]
            ),
            False,
        ),
        (
            "status_error",
            "config_1",
            HTTPStatus.NOT_FOUND,
            {"error": "error"},
            CachedArtifactError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            "config_1",
            HTTPStatus.OK,
            {"not_parquet_files": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
        (
            "shards_order",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetAndInfoResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="parquet-train-00001-of-05534.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="parquet-train-00000-of-05534.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="test",
                        url="url2",
                        filename="parquet-test-00000-of-00001.parquet",
                        size=0,
                    ),
                ],
                dataset_info={"description": "value", "dataset_size": 10},
            ),
            None,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="test",
                        url="url2",
                        filename="parquet-test-00000-of-00001.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="parquet-train-00000-of-05534.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="parquet-train-00001-of-05534.parquet",
                        size=0,
                    ),
                ]
            ),
            False,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="config-parquet-and-info",
        dataset=dataset,
        config=config,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = "doesnotexist"
    job_runner = get_job_runner(dataset, config, app_config)
    with pytest.raises(CachedArtifactError):
        job_runner.compute()
