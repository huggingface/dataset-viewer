# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable

import pytest
from libcommon.dataset import DatasetNotFoundError
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runners.config.split_names_from_dataset_info import (
    PreviousStepFormatError,
    PreviousStepStatusError,
    SplitNamesFromDatasetInfoJobRunner,
)
from worker.resources import LibrariesResource

GetJobRunner = Callable[[str, str, AppConfig, bool], SplitNamesFromDatasetInfoJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> SplitNamesFromDatasetInfoJobRunner:
        return SplitNamesFromDatasetInfoJobRunner(
            job_info={
                "type": SplitNamesFromDatasetInfoJobRunner.get_job_type(),
                "dataset": dataset,
                "config": config,
                "split": None,
                "job_id": "job_id",
                "force": force,
                "priority": Priority.NORMAL,
            },
            app_config=app_config,
            processing_step=ProcessingStep(
                name=SplitNamesFromDatasetInfoJobRunner.get_job_type(),
                input_type="config",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
                job_runner_version=SplitNamesFromDatasetInfoJobRunner.get_job_runner_version(),
            ),
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
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
                    "config_name": {
                        "splits": {
                            "train": {"name": "train", "dataset_name": "ok"},
                            "validation": {"name": "validation", "dataset_name": "ok"},
                            "test": {"name": "test", "dataset_name": "ok"},
                        },
                    }
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
            PreviousStepStatusError.__name__,
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
    upsert_response(kind="/dataset-info", dataset=dataset, content=upstream_content, http_status=upstream_status)
    job_runner = get_job_runner(dataset, "config_name", app_config, False)
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
    assert exc_info.value.code == DatasetNotFoundError.__name__
