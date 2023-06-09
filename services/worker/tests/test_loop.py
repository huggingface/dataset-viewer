from dataclasses import replace

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runner import JobRunner
from worker.job_runner_factory import BaseJobRunnerFactory
from worker.loop import Loop
from worker.resources import LibrariesResource
from worker.utils import CompleteJobResult


class DummyJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dummy"

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


class DummyJobRunnerFactory(BaseJobRunnerFactory):
    def __init__(
        self, processing_graph: ProcessingGraph, processing_step: ProcessingStep, app_config: AppConfig
    ) -> None:
        self.processing_step = processing_step
        self.processing_graph = processing_graph
        self.app_config = app_config

    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        return DummyJobRunner(
            job_info=job_info,
            app_config=self.app_config,
            processing_step=self.processing_step,
        )


def test_process_next_job(
    test_processing_graph: ProcessingGraph,
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    worker_state_file_path: str,
) -> None:
    job_type = test_processing_step.job_type
    app_config = replace(app_config, worker=replace(app_config.worker, job_types_only=[job_type]))

    factory = DummyJobRunnerFactory(
        processing_step=test_processing_step, processing_graph=test_processing_graph, app_config=app_config
    )

    loop = Loop(
        job_runner_factory=factory,
        library_cache_paths=libraries_resource.storage_paths,
        app_config=app_config,
        state_file_path=worker_state_file_path,
        processing_graph=test_processing_graph,
    )
    assert not loop.process_next_job()
    dataset = "dataset"
    revision = "revision"
    config = "config"
    split = "split"
    loop.queue.upsert_job(job_type=job_type, dataset=dataset, revision=revision, config=config, split=split)
    assert loop.queue.is_job_in_process(
        job_type=job_type, dataset=dataset, revision=revision, config=config, split=split
    )
    assert loop.process_next_job()
    assert not loop.queue.is_job_in_process(
        job_type=job_type, dataset=dataset, revision=revision, config=config, split=split
    )
