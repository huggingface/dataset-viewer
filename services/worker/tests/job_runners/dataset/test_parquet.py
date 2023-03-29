# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable

import pytest
from libcommon.dataset import DatasetNotFoundError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runners.config.parquet import ConfigParquetResponse
from worker.job_runners.config.parquet_and_info import ParquetFileItem
from worker.job_runners.dataset.parquet import (
    DatasetParquetJobRunner,
    DatasetParquetResponse,
    PreviousStepFormatError,
    PreviousStepStatusError,
)

from ..utils import UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig, bool], DatasetParquetJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> DatasetParquetJobRunner:
        return DatasetParquetJobRunner(
            job_info={
                "type": DatasetParquetJobRunner.get_job_type(),
                "dataset": dataset,
                "config": None,
                "split": None,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            common_config=app_config.common,
            worker_config=app_config.worker,
            processing_step=ProcessingStep(
                name=DatasetParquetJobRunner.get_job_type(),
                input_type="dataset",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
                job_runner_version=DatasetParquetJobRunner.get_job_runner_version(),
            ),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            [
                UpstreamResponse(
                    kind="/config-names",
                    dataset="ok",
                    config=None,
                    http_status=HTTPStatus.OK,
                    content={
                        "config_names": [
                            {"dataset": "dataset_ok", "config": "config_1"},
                            {"dataset": "dataset_ok", "config": "config_2"},
                        ],
                    },
                ),
                UpstreamResponse(
                    kind="config-parquet",
                    dataset="ok",
                    config="config_1",
                    http_status=HTTPStatus.OK,
                    content=ConfigParquetResponse(
                        parquet_files=[
                            ParquetFileItem(
                                dataset="ok",
                                config="config_1",
                                split="train",
                                url="url1",
                                filename="filename1",
                                size=0,
                            ),
                        ]
                    ),
                ),
                UpstreamResponse(
                    kind="config-parquet",
                    dataset="ok",
                    config="config_2",
                    http_status=HTTPStatus.OK,
                    content=ConfigParquetResponse(
                        parquet_files=[
                            ParquetFileItem(
                                dataset="ok",
                                config="config_2",
                                split="train",
                                url="url2",
                                filename="filename2",
                                size=0,
                            ),
                        ]
                    ),
                ),
            ],
            None,
            DatasetParquetResponse(
                parquet_files=[
                    ParquetFileItem(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    ParquetFileItem(
                        dataset="ok", config="config_2", split="train", url="url2", filename="filename2", size=0
                    ),
                ],
                pending=[],
                failed=[],
            ),
            False,
        ),
        (
            "status_error",
            [
                UpstreamResponse(
                    kind="/config-names",
                    dataset="status_error",
                    config=None,
                    http_status=HTTPStatus.NOT_FOUND,
                    content={"error": "error"},
                )
            ],
            PreviousStepStatusError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            [
                UpstreamResponse(
                    kind="/config-names",
                    dataset="format_error",
                    config=None,
                    http_status=HTTPStatus.OK,
                    content={"not_parquet_files": "wrong_format"},
                )
            ],
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expected_error_code: str,
    expected_content: Any,
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
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config, False)
    with pytest.raises(DatasetNotFoundError):
        job_runner.compute()
