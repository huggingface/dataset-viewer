# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable
from unittest.mock import Mock

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph
from libcommon.utils import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.common_exceptions import PreviousStepError
from worker.config import AppConfig
from worker.job_operators.config.split_names_from_dataset_info import (
    PreviousStepFormatError,
    SplitNamesFromDatasetInfoJobOperator,
)

GetJobRunner = Callable[[str, str, AppConfig, bool], SplitNamesFromDatasetInfoJobOperator]


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
    ) -> SplitNamesFromDatasetInfoJobRunner:
        processing_step_name = SplitNamesFromDatasetInfoJobRunner.get_job_type()
        processing_graph = ProcessingGraph(
            {
                "dataset-level": {"input_type": "dataset"},
                processing_step_name: {
                    "input_type": "dataset",
                    "job_runner_version": SplitNamesFromDatasetInfoJobRunner.get_job_runner_version(),
                    "triggered_by": "dataset-level",
                },
            }
        )
        return SplitNamesFromDatasetInfoJobRunner(
            job_info={
                "type": SplitNamesFromDatasetInfoJobOperator.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": config,
                    "split": None,
                    "git_revision": "1.0",
                },
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=processing_graph.get_processing_step(processing_step_name),
            processing_graph=processing_graph,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_status,upstream_content,error_code,content",
    [
        (
            "ok",
            HTTPStatus.OK,
            {
                "dataset_info": {
                    "splits": {
                        "train": {"name": "train", "dataset_name": "ok"},
                        "validation": {"name": "validation", "dataset_name": "ok"},
                        "test": {"name": "test", "dataset_name": "ok"},
                    },
                }
            },
            None,
            {
                "splits": [
                    {"dataset": "ok", "config": "config_name", "split": "train"},
                    {"dataset": "ok", "config": "config_name", "split": "validation"},
                    {"dataset": "ok", "config": "config_name", "split": "test"},
                ]
            },
        ),
        (
            "upstream_fail",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"error": "error"},
            PreviousStepError.__name__,
            None,
        ),
        (
            "without_dataset_info",
            HTTPStatus.OK,
            {"some_column": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
        ),
        (
            "without_config_name",
            HTTPStatus.OK,
            {"dataset_info": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
        ),
        (
            "without_splits",
            HTTPStatus.OK,
            {"dataset_info": {"config_name": "wrong_format"}},
            PreviousStepFormatError.__name__,
            None,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    error_code: str,
    content: Any,
) -> None:
    config = "config_name"
    upsert_response(
        kind="config-info", dataset=dataset, config=config, content=upstream_content, http_status=upstream_status
    )
    job_runner = get_job_runner(dataset, config, app_config, False)
    job_runner.get_dataset_git_revision = Mock(return_value="1.0.0")  # type: ignore

    if error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.type.__name__ == error_code
    else:
        assert job_runner.compute().content == content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "non_existent"
    config = "non_existent"
    worker = get_job_runner(dataset, config, app_config, False)
    with pytest.raises(CustomError) as exc_info:
        worker.compute()
    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND
    assert exc_info.value.code == "CachedResponseNotFound"
