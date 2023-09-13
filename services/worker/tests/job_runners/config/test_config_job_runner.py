# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Optional

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.config.config_job_runner import ConfigJobRunner


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


class DummyConfigJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


def test_failed_creation(test_processing_step: ProcessingStep, app_config: AppConfig) -> None:
    with pytest.raises(CustomError) as exc_info:
        DummyConfigJobRunner(
            job_info={
                "job_id": "job_id",
                "type": test_processing_step.job_type,
                "params": {
                    "dataset": "dataset",
                    "revision": "revision",
                    "config": None,
                    "split": None,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
    assert exc_info.value.code == "ParameterMissingError"


@pytest.mark.parametrize(
    "upsert_config,exception_name",
    [
        ("config", None),
        ("other_config", "ConfigNotFoundError"),
    ],
)
def test_creation(
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
    upsert_config: str,
    exception_name: Optional[str],
) -> None:
    dataset, config = "dataset", "config"

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        content={"config_names": [{"dataset": dataset, "config": upsert_config}]},
        http_status=HTTPStatus.OK,
    )

    if exception_name is None:
        assert (
            DummyConfigJobRunner(
                job_info={
                    "job_id": "job_id",
                    "type": test_processing_step.job_type,
                    "params": {
                        "dataset": dataset,
                        "revision": "revision",
                        "config": config,
                        "split": None,
                    },
                    "priority": Priority.NORMAL,
                    "difficulty": 50,
                },
                processing_step=test_processing_step,
                app_config=app_config,
            )
            is not None
        )
    else:
        with pytest.raises(CustomError) as exc_info:
            DummyConfigJobRunner(
                job_info={
                    "job_id": "job_id",
                    "type": test_processing_step.job_type,
                    "params": {
                        "dataset": dataset,
                        "revision": "revision",
                        "config": config,
                        "split": None,
                    },
                    "priority": Priority.NORMAL,
                    "difficulty": 50,
                },
                processing_step=test_processing_step,
                app_config=app_config,
            )
        assert exc_info.value.code == exception_name
