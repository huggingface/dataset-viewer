# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Mapping, Optional

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from libcommon.simple_cache import SplitFullName

from worker.config import AppConfig
from worker.job_runner import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunner


class DummySplitJobRunner(SplitJobRunner):
    def get_dataset_git_revision(self) -> Optional[str]:
        return "0.0.1"

    @staticmethod
    def _get_dataset_git_revision() -> Optional[str]:
        return "0.0.1"

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        return {SplitFullName(self.dataset, self.config, self.split)}


@pytest.mark.parametrize("config,split", [(None, None), (None, "split"), ("config", None)])
def test_failed_creation(test_processing_step: ProcessingStep, app_config: AppConfig, config: str, split: str) -> None:
    with pytest.raises(CustomError) as exc_info:
        DummySplitJobRunner(
            job_info={
                "job_id": "job_id",
                "type": test_processing_step.job_type,
                "dataset": "dataset",
                "config": config,
                "split": split,
                "force": False,
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
    assert exc_info.value.code == "ParameterMissingError"
    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST


def test_success_creation(test_processing_step: ProcessingStep, app_config: AppConfig) -> None:
    assert (
        DummySplitJobRunner(
            job_info={
                "job_id": "job_id",
                "type": test_processing_step.job_type,
                "dataset": "dataset",
                "config": "config",
                "split": "split",
                "force": False,
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
        is not None
    )
