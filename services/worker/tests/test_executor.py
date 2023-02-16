import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytz
from filelock import FileLock
from libcommon.queue import Job, JobInfo, Priority, Status, get_datetime
from libcommon.resources import QueueMongoResource
from pytest import fixture

from worker.config import AppConfig
from worker.loop import WorkerState
from worker.main import WorkerExecutor


def get_job_info() -> JobInfo:
    return JobInfo(
        job_id="a" * 24,
        type="bar",
        dataset="user/my_dataset",
        config="default",
        split="train",
        force=False,
        priority=Priority.LOW,
    )


def write_worker_state(worker_state: WorkerState, worker_state_path: str) -> None:
    with FileLock(worker_state_path + ".lock"):
        with open(worker_state_path, "w") as worker_state_f:
            json.dump(worker_state, worker_state_f)


def start_worker_loop() -> None:
    app_config = AppConfig.from_env()
    if not app_config.worker.state_path:
        raise ValueError("Failed to get worker state because WORKER_STATE_PATH is missing.")
    if "--print-worker-state-path" in sys.argv:
        print(app_config.worker.state_path, flush=True)
    current_job_info = get_job_info()
    worker_state = WorkerState(current_job_info=current_job_info)
    write_worker_state(worker_state, app_config.worker.state_path)


@fixture
def set_worker_state(worker_state_path: Path) -> Iterator[WorkerState]:
    job_info = get_job_info()
    worker_state = WorkerState(current_job_info=job_info)
    write_worker_state(worker_state, str(worker_state_path))
    yield worker_state
    os.remove(worker_state_path)


@fixture
def set_started_job_in_queue(queue_mongo_resource: QueueMongoResource) -> Iterator[Job]:
    if not queue_mongo_resource.is_available():
        raise RuntimeError("Mongo resource is not available")
    job_info = get_job_info()
    if Job.objects.with_id(job_info["job_id"]):  # type: ignore
        Job.objects.with_id(job_info["job_id"]).delete()  # type: ignore
    job = Job(
        pk=job_info["job_id"],
        type=job_info["type"],
        dataset=job_info["dataset"],
        config=job_info["config"],
        split=job_info["split"],
        unicity_id="unicity_id",
        namespace="user",
        priority=job_info["priority"],
        status=Status.STARTED,
        created_at=get_datetime(),
    )
    job.save()
    yield job
    job.delete()


def test_executor_get_state(app_config: AppConfig, set_worker_state: WorkerState) -> None:
    executor = WorkerExecutor(app_config)
    assert executor.get_state() == set_worker_state


def test_executor_get_empty_state(app_config: AppConfig) -> None:
    executor = WorkerExecutor(app_config)
    assert executor.get_state() == WorkerState(current_job_info=None)


def test_executor_get_current_job(
    app_config: AppConfig, set_started_job_in_queue: Job, set_worker_state: WorkerState
) -> None:
    executor = WorkerExecutor(app_config)
    assert executor.get_current_job() == set_started_job_in_queue


def test_executor_get_nonexisting_current_job(app_config: AppConfig) -> None:
    executor = WorkerExecutor(app_config)
    assert executor.get_current_job() is None


def test_executor_heartbeat(
    app_config: AppConfig,
    set_started_job_in_queue: Job,
    set_worker_state: WorkerState,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    if not queue_mongo_resource.is_available():
        raise RuntimeError("Mongo resource is not available")
    executor = WorkerExecutor(app_config)
    current_job = executor.get_current_job()
    assert current_job is not None
    assert current_job.last_heartbeat is None
    executor.heartbeat()
    current_job = executor.get_current_job()
    assert current_job is not None
    assert current_job.last_heartbeat is not None
    last_heartbeat_datetime = pytz.UTC.localize(current_job.last_heartbeat)
    assert last_heartbeat_datetime >= get_datetime() - timedelta(seconds=1)


def test_executor_start(app_config: AppConfig, queue_mongo_resource: QueueMongoResource) -> None:
    if not queue_mongo_resource.is_available():
        raise RuntimeError("Mongo resource is not available")
    executor = WorkerExecutor(app_config)
    with patch.object(executor, "heartbeat", wraps=executor.heartbeat) as heartbeat_mock:
        with patch("worker.main.START_WORKER_LOOP_PATH", __file__):
            executor.start()
    current_job = executor.get_current_job()
    assert current_job is not None
    assert current_job.pk == get_job_info()["job_id"]
    assert heartbeat_mock.call_count > 0


if __name__ == "__main__":
    start_worker_loop()
