from typing import Any, Mapping, Optional

from libcommon.config import CommonConfig, QueueConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from libcommon.resources import CacheDatabaseResource, QueueDatabaseResource

from datasets_based.config import DatasetsBasedConfig, WorkerLoopConfig
from datasets_based.resources import LibrariesResource
from datasets_based.worker import JobInfo, Worker
from datasets_based.worker_factory import BaseWorkerFactory
from datasets_based.worker_loop import WorkerLoop


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


class DummyWorkerFactory(BaseWorkerFactory):
    def __init__(self, processing_step: ProcessingStep) -> None:
        self.common_config = CommonConfig()
        self.datasets_based_config = DatasetsBasedConfig()
        self.processing_step = processing_step

    def _create_worker(self, job_info: JobInfo) -> Worker:
        return DummyWorker(
            job_info=job_info,
            common_config=self.common_config,
            datasets_based_config=self.datasets_based_config,
            processing_step=self.processing_step,
        )


def test_process_next_job(
    test_processing_step: ProcessingStep,
    queue_config: QueueConfig,
    libraries_resource: LibrariesResource,
    cache_database_resource: CacheDatabaseResource,
    queue_database_resource: QueueDatabaseResource,
) -> None:
    worker_factory = DummyWorkerFactory(processing_step=test_processing_step)
    queue = Queue(type=test_processing_step.endpoint, max_jobs_per_namespace=queue_config.max_jobs_per_namespace)
    worker_loop = WorkerLoop(
        library_cache_paths=libraries_resource.storage_paths,
        queue=queue,
        worker_factory=worker_factory,
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
