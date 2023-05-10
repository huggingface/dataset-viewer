from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Mapping, Optional
from unittest.mock import Mock

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedResponse,
    DoesNotExist,
    SplitFullName,
    get_response,
    get_response_with_details,
    upsert_response,
)
from libcommon.utils import JobInfo, Priority, Status

from worker.common_exceptions import PreviousStepError
from worker.config import AppConfig
from worker.job_manager import ERROR_CODES_TO_RETRY, JobManager
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import CompleteJobResult
from .fixtures.hub import get_default_config_split


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


class DummyJobRunner(DatasetJobRunner):
    # override get_dataset_git_revision to avoid making a request to the Hub
    def get_dataset_git_revision(self) -> Optional[str]:
        return DummyJobRunner._get_dataset_git_revision()

    @staticmethod
    def _get_dataset_git_revision() -> Optional[str]:
        return "0.1.2"

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        return {SplitFullName(self.dataset, "config", "split1"), SplitFullName(self.dataset, "config", "split2")}


def test_check_type(
    test_processing_graph: ProcessingGraph,
    another_processing_step: ProcessingStep,
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    force = False

    job_type = f"not-{test_processing_step.job_type}"
    with pytest.raises(ValueError):
        DummyJobRunner(
            job_info={
                "job_id": job_id,
                "type": job_type,
                "params": {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                },
                "force": force,
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            app_config=app_config,
        )
    with pytest.raises(ValueError):
        DummyJobRunner(
            job_info={
                "job_id": job_id,
                "type": test_processing_step.job_type,
                "params": {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                },
                "force": force,
                "priority": Priority.NORMAL,
            },
            processing_step=another_processing_step,
            app_config=app_config,
        )
