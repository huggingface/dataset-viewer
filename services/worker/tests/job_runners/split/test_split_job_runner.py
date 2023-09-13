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
from worker.job_runners.split.split_job_runner import SplitJobRunner


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


class DummySplitJobRunner(SplitJobRunner):
    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


@pytest.mark.parametrize("config,split", [(None, None), (None, "split"), ("config", None)])
def test_failed_creation(test_processing_step: ProcessingStep, app_config: AppConfig, config: str, split: str) -> None:
    upsert_response(
        kind="dataset-config-names",
        dataset="dataset",
        content={"config_names": [{"dataset": "dataset", "config": config}]},
        http_status=HTTPStatus.OK,
    )

    with pytest.raises(CustomError) as exc_info:
        DummySplitJobRunner(
            job_info={
                "job_id": "job_id",
                "type": test_processing_step.job_type,
                "params": {
                    "dataset": "dataset",
                    "revision": "revision",
                    "config": config,
                    "split": split,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
    assert exc_info.value.code == "ParameterMissingError"


@pytest.mark.parametrize(
    "upsert_config,upsert_split,exception_name",
    [
        ("config", "split", None),
        ("config", "other_split", "SplitNotFoundError"),
        ("other_config", "split", "ConfigNotFoundError"),
    ],
)
def test_creation(
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
    upsert_config: str,
    upsert_split: str,
    exception_name: Optional[str],
) -> None:
    dataset, config, split = "dataset", "config", "split"

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        content={"config_names": [{"dataset": dataset, "config": upsert_config}]},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind="config-split-names-from-streaming",
        dataset=dataset,
        config=config,
        content={"splits": [{"dataset": dataset, "config": upsert_config, "split": upsert_split}]},
        http_status=HTTPStatus.OK,
    )

    if exception_name is None:
        assert (
            DummySplitJobRunner(
                job_info={
                    "job_id": "job_id",
                    "type": test_processing_step.job_type,
                    "params": {
                        "dataset": dataset,
                        "revision": "revision",
                        "config": config,
                        "split": split,
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
            DummySplitJobRunner(
                job_info={
                    "job_id": "job_id",
                    "type": test_processing_step.job_type,
                    "params": {
                        "dataset": dataset,
                        "revision": "revision",
                        "config": config,
                        "split": split,
                    },
                    "priority": Priority.NORMAL,
                    "difficulty": 50,
                },
                processing_step=test_processing_step,
                app_config=app_config,
            )
        assert exc_info.value.code == exception_name
