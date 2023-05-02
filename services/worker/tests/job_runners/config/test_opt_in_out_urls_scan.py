# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable, List

import pytest
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runner import PreviousStepError
from worker.job_runners.config.opt_in_out_urls_scan_from_streaming import (
    ConfigOptInOutUrlsScanJobRunner,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig, bool], ConfigOptInOutUrlsScanJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> ConfigOptInOutUrlsScanJobRunner:
        return ConfigOptInOutUrlsScanJobRunner(
            job_info={
                "type": ConfigOptInOutUrlsScanJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": None,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            common_config=app_config.common,
            worker_config=app_config.worker,
            processing_step=ProcessingStep(
                name=ConfigOptInOutUrlsScanJobRunner.get_job_type(),
                input_type="config",
                requires=[],
                required_by_dataset_viewer=False,
                ancestors=[],
                children=[],
                parents=[],
                job_runner_version=ConfigOptInOutUrlsScanJobRunner.get_job_runner_version(),
            ),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,split_names_status,split_names_content,spawning_status"
    + ",spawning_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            "config",
            HTTPStatus.OK,
            {
                "splits": [
                    {"dataset": "dataset_ok", "config": "config", "split": "split"},
                    {"dataset": "dataset_ok", "config": "config", "split": "split2"},
                ]
            },
            [HTTPStatus.OK, HTTPStatus.OK],
            [
                {
                    "urls_columns": ["url"],
                    "num_opt_in_urls": 1,
                    "num_opt_out_urls": 2,
                    "num_urls": 10,
                    "num_scanned_rows": 100,
                    "has_urls_columns": True,
                },
                {
                    "urls_columns": [],
                    "num_opt_in_urls": 0,
                    "num_opt_out_urls": 0,
                    "num_urls": 0,
                    "num_scanned_rows": 30,
                    "has_urls_columns": False,
                },
            ],
            None,
            {
                "urls_columns": ["url"],
                "num_opt_in_urls": 1,
                "num_opt_out_urls": 2,
                "num_urls": 10,
                "num_scanned_rows": 130,
                "has_urls_columns": True,
            },
            False,
        ),
        (
            "previos_step_error",
            "config",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {},
            [],
            [],
            "PreviousStepError",
            None,
            True,
        ),
        (
            "previous_step_format_error",
            "config",
            HTTPStatus.OK,
            {
                "splits": [
                    {"dataset": "dataset_ok", "config": "config", "split": "split"},
                    {"dataset": "dataset_ok", "config": "config", "split": "split2"},
                ]
            },
            [HTTPStatus.OK],
            [{"wrong_format": None}],
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
    split_names_status: HTTPStatus,
    split_names_content: Any,
    spawning_status: List[HTTPStatus],
    spawning_content: List[Any],
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=dataset,
        config=config,
        content=split_names_content,
        http_status=split_names_status,
    )

    if split_names_status == HTTPStatus.OK:
        for split_item, status, content in zip(split_names_content["splits"], spawning_status, spawning_content):
            upsert_response(
                kind="split-opt-in-out-urls-scan",
                dataset=dataset,
                config=split_item["config"],
                split=split_item["split"],
                content=content,
                http_status=status,
            )

    job_runner = get_job_runner(dataset, config, app_config, False)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.type.__name__ == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = "doesnotexist"
    job_runner = get_job_runner(dataset, config, app_config, False)
    with pytest.raises(PreviousStepError):
        job_runner.compute()
