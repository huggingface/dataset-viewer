from typing import Any, Mapping, Optional

import pytest
from libcommon.config import CommonConfig, QueueConfig, WorkerLoopConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue

from datasets_based.config import AppConfig
from datasets_based.worker import JobInfo, Worker, WorkerFactory
from datasets_based.worker_loop import WorkerLoop


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


class DummyWorker(Worker):
    # override get_dataset_git_revision to avoid making a request to the Hub
    def get_dataset_git_revision(self) -> Optional[str]:
        return "0.1.2"

    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    @staticmethod
    def get_version() -> str:
        return "1.0.1"

    def compute(self) -> Mapping[str, Any]:
        return {"key": "value"}


class DummyWorkerFactory(WorkerFactory):
    def __init__(self, processing_step: ProcessingStep) -> None:
        self.common_config = CommonConfig()
        self.processing_step = processing_step

    def _create_worker(self, job_info: JobInfo) -> Worker:
        return DummyWorker(job_info=job_info, common_config=self.common_config, processing_step=self.processing_step)


def test_process_next_job(
    test_processing_step: ProcessingStep,
    queue_config: QueueConfig,
) -> None:
    worker_factory = DummyWorkerFactory(processing_step=test_processing_step)
    queue = Queue(type=test_processing_step.endpoint, max_jobs_per_namespace=queue_config.max_jobs_per_namespace)
    worker_loop = WorkerLoop(
        worker_factory=worker_factory,
        queue=queue,
        worker_loop_config=WorkerLoopConfig(),
    )
    assert worker_loop.process_next_job() is False
    dataset = "dataset"
    config = "config"
    split = "split"
    worker_loop.queue.upsert_job(dataset=dataset, config=config, split=split)
    worker_loop.queue.is_job_in_process(dataset=dataset, config=config, split=split) is True
    assert worker_loop.process_next_job() is True
    worker_loop.queue.is_job_in_process(dataset=dataset, config=config, split=split) is False
