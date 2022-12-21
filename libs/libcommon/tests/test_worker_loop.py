from typing import Any, Mapping, Optional

import pytest

from libcommon.config import CommonConfig, QueueConfig, WorkerLoopConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import _clean_cache_database
from libcommon.worker import Worker
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


class NoStorageWorkerLoop(WorkerLoop):
    def has_storage(self) -> bool:
        return False


def test_has_storage(
    test_processing_step: ProcessingStep,
    common_config: CommonConfig,
    queue_config: QueueConfig,
    worker_loop_config: WorkerLoopConfig,
) -> None:
    worker_loop = WorkerLoop(
        worker_class=DummyWorker,
        processing_step=test_processing_step,
        common_config=common_config,
        queue_config=queue_config,
        worker_loop_config=worker_loop_config,
    )
    assert worker_loop.has_storage() is True
    worker_loop = NoStorageWorkerLoop(
        worker_class=DummyWorker,
        processing_step=test_processing_step,
        common_config=common_config,
        queue_config=queue_config,
        worker_loop_config=worker_loop_config,
    )
    assert worker_loop.has_storage() is False
