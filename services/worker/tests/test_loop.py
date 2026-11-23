from collections.abc import Iterator
from dataclasses import replace

from huggingface_hub import HfFileSystem
from libcommon.dtos import JobInfo
from libcommon.resources import CacheMongoResource, QueueMongoResource
from pytest import MonkeyPatch

from worker import loop as loop_module
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

    def compute(self) -> Iterator[CompleteJobResult]:
        yield CompleteJobResult({"key": "value"})


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


def make_loop(
    app_config: AppConfig,
    worker_state_file_path: str,
    max_seconds_blocked_by_pod_memory: float,
    sleep_seconds: float = 0,
) -> Loop:
    worker = replace(
        app_config.worker,
        max_seconds_blocked_by_pod_memory=max_seconds_blocked_by_pod_memory,
        sleep_seconds=sleep_seconds,
    )
    config = replace(app_config, worker=worker)
    return Loop(
        job_runner_factory=DummyJobRunnerFactory(app_config=config),
        app_config=config,
        state_file_path=worker_state_file_path,
    )


def test_pod_memory_block_frees_hf_file_system_cache_then_waits(
    app_config: AppConfig, queue_mongo_resource: QueueMongoResource, worker_state_file_path: str
) -> None:
    loop = make_loop(app_config, worker_state_file_path, max_seconds_blocked_by_pod_memory=60)
    HfFileSystem()  # fills the instance cache, like a job reading from the Hub
    assert HfFileSystem._cache

    assert not loop.should_exit_for_pod_memory()
    assert not HfFileSystem._cache
    assert not loop.should_exit_for_pod_memory()


def test_run_exits_when_pod_memory_stays_too_high(
    app_config: AppConfig, queue_mongo_resource: QueueMongoResource, worker_state_file_path: str
) -> None:
    loop = make_loop(app_config, worker_state_file_path, max_seconds_blocked_by_pod_memory=0)
    loop.has_pod_memory = lambda: False  # type: ignore[method-assign]
    loop.queue.add_job(job_type=JOB_TYPE, dataset="dataset", revision="revision", difficulty=50)

    loop.run()  # returns instead of sleeping forever

    assert loop.queue.is_job_in_process(job_type=JOB_TYPE, dataset="dataset", revision="revision")


def test_pod_memory_block_resets_once_memory_is_back_under_the_limit(
    app_config: AppConfig,
    queue_mongo_resource: QueueMongoResource,
    worker_state_file_path: str,
    monkeypatch: MonkeyPatch,
) -> None:
    frees: list[None] = []
    monkeypatch.setattr(loop_module, "free_memory", lambda: frees.append(None))
    loop = make_loop(app_config, worker_state_file_path, max_seconds_blocked_by_pod_memory=0)
    pod_memory_ok = iter([False, True, False])
    loop.has_pod_memory = lambda: next(pod_memory_ok, False)  # type: ignore[method-assign]

    loop.run()

    # memory was freed again for the second block instead of exiting on the first block's timer
    assert len(frees) == 2


def test_frees_memory_after_a_job_only_above_the_soft_limit(
    app_config: AppConfig,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    worker_state_file_path: str,
    monkeypatch: MonkeyPatch,
) -> None:
    frees: list[None] = []
    monkeypatch.setattr(loop_module, "free_memory", lambda: frees.append(None))
    loop = make_loop(app_config, worker_state_file_path, max_seconds_blocked_by_pod_memory=60)
    loop._get_pod_memory_limit_bytes = lambda: 10 * 1024**3  # type: ignore[method-assign]

    loop._get_pod_memory_usage_bytes = lambda: 4 * 1024**3  # type: ignore[method-assign]  # 40%
    loop.queue.add_job(job_type=JOB_TYPE, dataset="dataset", revision="revision", difficulty=50)
    assert loop.process_next_job()
    assert not frees
    loop.queue.delete_dataset_waiting_jobs("dataset")  # the follow-up jobs the first job created

    loop._get_pod_memory_usage_bytes = lambda: 6 * 1024**3  # type: ignore[method-assign]  # 60%
    loop.queue.add_job(job_type=JOB_TYPE, dataset="dataset", revision="revision2", difficulty=50)
    assert loop.process_next_job()
    assert len(frees) == 1
