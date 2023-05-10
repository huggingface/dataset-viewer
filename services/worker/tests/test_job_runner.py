from typing import Optional

import pytest
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.utils import Priority

from worker.config import AppConfig
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import CompleteJobResult


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
        return "dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


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
