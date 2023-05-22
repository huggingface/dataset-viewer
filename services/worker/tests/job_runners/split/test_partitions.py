# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.split.partitions import SplitPartitionsJobRunner


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, str, AppConfig], SplitPartitionsJobRunner]


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
    ) -> SplitPartitionsJobRunner:
        processing_step_name = SplitPartitionsJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                "config-level": {"input_type": "config", "triggered_by": "dataset-level"},
                processing_step_name: {
                    "input_type": "split",
                    "job_runner_version": SplitPartitionsJobRunner.get_job_runner_version(),
                    "triggered_by": "config-level",
                },
            }
        )
        return SplitPartitionsJobRunner(
            job_info={
                "type": SplitPartitionsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                    "partition": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,split,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            "config_ok",
            "split_ok",
            HTTPStatus.OK,
            {
                "size": {
                    "splits": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_ok",
                            "split": "split_ok",
                            "num_bytes_parquet_files": 14_281_188,
                            "num_bytes_memory": 17_470_800,
                            "num_rows": 300_010,
                            "num_columns": 2,
                        },
                    ],
                }
            },
            None,
            {
                "num_rows": 300_010,
                "partitions": [
                    {
                        "partition": "0-99999",
                    },
                    {
                        "partition": "100000-199999",
                    },
                    {
                        "partition": "200000-299999",
                    },
                    {
                        "partition": "300000-300009",
                    },
                ],
            },
            False,
        ),
        (
            "dataset_previous_step_error",
            "config_previous_step_error",
            "split_previous_step_error",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {},
            "CachedArtifactError",
            None,
            True,
        ),
        (
            "dataset_format_error",
            "config_format_error",
            "split_format_error",
            HTTPStatus.OK,
            {"wrong_format": None},
            "PreviousStepFormatError",
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: str,
    split: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="config-size",
        dataset=dataset,
        config=config,
        split=None,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, split, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content
