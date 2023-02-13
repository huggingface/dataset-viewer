from typing import Any, Mapping, Optional

from libcommon.config import CommonConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource

from worker.config import AppConfig, WorkerConfig
from worker.job_runner import JobInfo, JobRunner
from worker.job_runner_factory import BaseJobRunnerFactory
from worker.loop import Loop
from worker.resources import LibrariesResource


class DummyJobRunner(JobRunner):
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


class DummyJobRunnerFactory(BaseJobRunnerFactory):
    def __init__(self, processing_step: ProcessingStep) -> None:
        self.common_config = CommonConfig()
        self.worker_config = WorkerConfig()
        self.processing_step = processing_step

    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        return DummyJobRunner(
            job_info=job_info,
            common_config=self.common_config,
            worker_config=self.worker_config,
            processing_step=self.processing_step,
        )


def test_process_next_job(
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    factory = DummyJobRunnerFactory(processing_step=test_processing_step)
    queue = Queue(type=test_processing_step.endpoint, max_jobs_per_namespace=app_config.queue.max_jobs_per_namespace)
    loop = Loop(
        library_cache_paths=libraries_resource.storage_paths,
        queue=queue,
        job_runner_factory=factory,
        worker_config=WorkerConfig(),
    )
    assert loop.process_next_job() is False
    dataset = "dataset"
    config = "config"
    split = "split"
    loop.queue.upsert_job(dataset=dataset, config=config, split=split)
    loop.queue.is_job_in_process(dataset=dataset, config=config, split=split) is True
    assert loop.process_next_job() is True
    loop.queue.is_job_in_process(dataset=dataset, config=config, split=split) is False
