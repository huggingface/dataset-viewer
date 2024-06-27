from libcommon.dtos import JobInfo
from libcommon.resources import CacheMongoResource, QueueMongoResource

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_runner import JobRunner
from worker.job_runner_factory import BaseJobRunnerFactory
from worker.loop import Loop
from worker.resources import LibrariesResource

JOB_TYPE = "dataset-config-names"


class DummyJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return JOB_TYPE

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


class DummyJobRunnerFactory(BaseJobRunnerFactory):
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        return DummyJobRunner(
            job_info=job_info,
            app_config=self.app_config,
        )


def test_process_next_job(
    app_config: AppConfig,
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    worker_state_file_path: str,
) -> None:
    factory = DummyJobRunnerFactory(app_config=app_config)

    loop = Loop(
        job_runner_factory=factory,
        app_config=app_config,
        state_file_path=worker_state_file_path,
    )
    assert not loop.process_next_job()
    dataset = "dataset"
    revision = "revision"
    loop.queue.add_job(job_type=JOB_TYPE, dataset=dataset, revision=revision, difficulty=50)
    assert loop.queue.is_job_in_process(job_type=JOB_TYPE, dataset=dataset, revision=revision)
    assert loop.process_next_job()
    assert not loop.queue.is_job_in_process(job_type=JOB_TYPE, dataset=dataset, revision=revision)
