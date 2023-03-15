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
from worker.job_runners.dataset.split_names_from_streaming import (
    DatasetSplitNamesFromStreamingJobRunner,
    PreviousStepFormatError,
)

GetJobRunner = Callable[[str, AppConfig, bool], DatasetSplitNamesFromStreamingJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
        force: bool = False,
    ) -> DatasetSplitNamesFromStreamingJobRunner:
        return DatasetSplitNamesFromStreamingJobRunner(
            job_info={
                "type": DatasetSplitNamesFromStreamingJobRunner.get_job_type(),
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
                name=DatasetSplitNamesFromStreamingJobRunner.get_job_type(),
                input_type="dataset",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
            ),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,split_names_from_streaming,expected_content,progress",
    [
        (
            "pending_response",
            [
                {
                    "config": "config_a",
                    "response": {
                        "splits": [
                            {
                                "dataset": "pending_response",
                                "config": "config_a",
                                "split": "split_a",
                            }
                        ]
                    },
                }
            ],
            {
                "splits": [
                    {
                        "dataset": "pending_response",
                        "config": "config_a",
                        "split": "split_a",
                    },
                ],
                "pending": [{"dataset": "pending_response", "config": "config_b"}],
                "failed": [],
            },
            0.5,
        ),
        (
            "complete",
            [
                {
                    "config": "config_a",
                    "response": {
                        "splits": [
                            {
                                "dataset": "complete",
                                "config": "config_a",
                                "split": "split_a",
                            }
                        ]
                    },
                },
                {
                    "config": "config_b",
                    "response": {
                        "splits": [
                            {
                                "dataset": "complete",
                                "config": "config_b",
                                "split": "split_b",
                            }
                        ]
                    },
                },
            ],
            {
                "splits": [
                    {
                        "dataset": "complete",
                        "config": "config_a",
                        "split": "split_a",
                    },
                    {
                        "dataset": "complete",
                        "config": "config_b",
                        "split": "split_b",
                    },
                ],
                "pending": [],
                "failed": [],
            },
            1,
        ),
    ],
)
def test_compute_progress(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    split_names_from_streaming: Any,
    expected_content: Any,
    progress: float,
) -> None:
    upsert_response(
        kind="/config-names",
        dataset=dataset,
        content={
            "config_names": [
                {
                    "dataset": dataset,
                    "config": "config_a",
                },
                {"dataset": dataset, "config": "config_b"},
            ]
        },
        http_status=HTTPStatus.OK,
    )
    for config in split_names_from_streaming:
        upsert_response(
            kind="/split-names-from-streaming",
            dataset=dataset,
            config=config["config"],
            content=config["response"],
            http_status=HTTPStatus.OK,
        )
    job_runner = get_job_runner(dataset, app_config, False)
    response = job_runner.compute()
    assert response.content == expected_content
    assert response.progress == progress


def test_compute_error(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "error"
    config = "error"
    upsert_response(
        kind="/config-names",
        dataset=dataset,
        content={
            "config_names": [
                {
                    "dataset": dataset,
                    "config": config,
                }
            ]
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    job_runner = get_job_runner(dataset, app_config, False)
    response = job_runner.compute()
    assert response.content == {
        "splits": [],
        "failed": [{"dataset": dataset, "config": config, "error": {}}],
        "pending": [],
    }
    assert response.progress == 1.0


def test_compute_format_error(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "error"
    config = "error"
    upsert_response(
        kind="/config-names",
        dataset=dataset,
        content={
            "config_names": [
                {
                    "dataset": dataset,
                    "config": config,
                }
            ]
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind="/split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"wrong_format": []},
        http_status=HTTPStatus.OK,
    )
    job_runner = get_job_runner(dataset, app_config, False)
    with pytest.raises(PreviousStepFormatError):
        job_runner.compute()


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config, False)
    with pytest.raises(DatasetNotFoundError):
        job_runner.compute()
