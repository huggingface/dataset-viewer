from typing import Any, Mapping, Optional

import pytest

from libcommon.config import CommonConfig, QueueConfig, WorkerLoopConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import _clean_cache_database
from libcommon.worker import StartedJobInfo, Worker, WorkerFactory
from libcommon.worker_loop import WorkerLoop


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_queue_database()
    _clean_cache_database()


class DummyWorker(Worker):
    # override get_dataset_git_revision to avoid making a request to the Hub
    def get_dataset_git_revision(self) -> Optional[str]:
        return "0.1.2"

    @staticmethod
    def get_endpoint() -> str:
        return "/dummy"

    @staticmethod
    def get_version() -> str:
        return "1.0.1"

    def compute(self) -> Mapping[str, Any]:
        return {"key": "value"}


class DummyWorkerFactory(WorkerFactory):
    def __init__(self, common_config: Any) -> None:
        self.common_config = common_config

    def _create_worker(self, started_job_info: StartedJobInfo, processing_step: ProcessingStep) -> Worker:
        return DummyWorker(
            started_job_info=started_job_info, common_config=self.common_config, processing_step=processing_step
        )


def test_process_next_job(
    test_processing_step: ProcessingStep,
    common_config: CommonConfig,
    queue_config: QueueConfig,
    worker_loop_config: WorkerLoopConfig,
) -> None:
    worker_factory = DummyWorkerFactory(common_config=common_config)
    worker_loop = WorkerLoop(
        worker_factory=worker_factory,
        processing_step=test_processing_step,
        queue_config=queue_config,
        worker_loop_config=worker_loop_config,
    )
    assert worker_loop.process_next_job() is False
    dataset = "dataset"
    config = "config"
    split = "split"
    worker_loop.queue.add_job(dataset=dataset, config=config, split=split)
    worker_loop.queue.is_job_in_process(dataset=dataset, config=config, split=split) is True
    assert worker_loop.process_next_job() is True
    worker_loop.queue.is_job_in_process(dataset=dataset, config=config, split=split) is False
