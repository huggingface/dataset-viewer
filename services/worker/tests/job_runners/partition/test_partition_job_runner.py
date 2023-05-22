# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.partition.partition_job_runner import PartitionJobRunner
from worker.utils import CompleteJobResult


class DummyPartitionJobRunner(PartitionJobRunner):
    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


@pytest.mark.parametrize(
    "config,split,partition",
    [
        (None, None, None),
        (None, "split", "0-100"),
        ("config", None, "0-100"),
        ("config", "split", "0-"),
        ("config", "split", "-100"),
        ("config", "split", "WRONG FORMAT"),
        ("config", "split", "-"),
    ],
)
def test_failed_creation(
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
    config: str,
    split: str,
    partition: str,
) -> None:
    with pytest.raises(CustomError) as exc_info:
        DummyPartitionJobRunner(
            job_info={
                "job_id": "job_id",
                "type": test_processing_step.job_type,
                "params": {
                    "dataset": "dataset",
                    "revision": "revision",
                    "config": config,
                    "split": split,
                    "partition": partition,
                },
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
    assert exc_info.value.code == "ParameterMissingError"
    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST


def test_success_creation(test_processing_step: ProcessingStep, app_config: AppConfig) -> None:
    assert (
        DummyPartitionJobRunner(
            job_info={
                "job_id": "job_id",
                "type": test_processing_step.job_type,
                "params": {
                    "dataset": "dataset",
                    "revision": "revision",
                    "config": "config",
                    "split": "split",
                    "partition": "0-100",
                },
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
        is not None
    )
