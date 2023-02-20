import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import timedelta
from http import HTTPStatus
from typing import Any, Callable, List, Optional

import pytz
from filelock import FileLock
from libcommon.exceptions import CustomError
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Job, JobInfo, Status, get_datetime
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import init_assets_dir
from mirakuru import OutputExecutor

from worker import start_worker_loop
from worker.config import AppConfig
from worker.job_runner_factory import JobRunnerFactory
from worker.loop import WorkerState
from worker.resources import LibrariesResource

WORKER_STATE_FILE_NAME = "worker_state.json"
START_WORKER_LOOP_PATH = start_worker_loop.__file__


async def every(
    func: Callable[..., Optional[Any]], *args: Any, seconds: int, stop_on: Optional[Any] = None, **kwargs: Any
) -> None:
    while True:
        out = func(*args, **kwargs)
        if stop_on is not None and out == stop_on:
            break
        await asyncio.sleep(seconds)


class WorkerCrashedError(CustomError):
    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "WorkerCrashedError", cause, True)


class WorkerExecutor:
    def __init__(self, app_config: AppConfig, job_runner_factory: JobRunnerFactory) -> None:
        self.app_config = app_config
        self.job_runner_factory = job_runner_factory

    def _create_worker_loop_executor(self) -> OutputExecutor:
        banner = self.app_config.worker.state_path
        if not banner:
            raise ValueError("Failed to create the executor because WORKER_STATE_PATH is missing.")
        start_worker_loop_command = [
            sys.executable,
            START_WORKER_LOOP_PATH,
            "--print-worker-state-path",
        ]
        return OutputExecutor(start_worker_loop_command, banner, timeout=10)

    def start(self) -> None:
        worker_loop_executor = self._create_worker_loop_executor()
        worker_loop_executor.start()  # blocking until the banner is printed

        loop = asyncio.get_event_loop()
        logging.info("Starting heartbeat.")
        loop.create_task(every(self.heartbeat, seconds=self.app_config.worker.heartbeat_time_interval_seconds))
        loop.create_task(every(self.kill_zombies, seconds=self.app_config.worker.kill_zombies_time_interval_seconds))
        loop.run_until_complete(
            every(self.is_worker_alive, worker_loop_executor=worker_loop_executor, seconds=1, stop_on=False)
        )

    def get_state(self) -> WorkerState:
        worker_state_path = self.app_config.worker.state_path
        if not worker_state_path:
            raise ValueError("Failed to get worker state because WORKER_STATE_PATH is missing.")
        if os.path.exists(worker_state_path):
            with FileLock(worker_state_path + ".lock"):
                try:
                    with open(worker_state_path, "r") as worker_state_f:
                        worker_state = json.load(worker_state_f)
                        return WorkerState(current_job_info=worker_state.get("current_job_info"))
                except json.JSONDecodeError:
                    return WorkerState(current_job_info=None)
        else:
            return WorkerState(current_job_info=None)

    def get_current_job(self) -> Optional[Job]:
        worker_state = self.get_state()
        if worker_state["current_job_info"]:
            job = Job.objects.with_id(worker_state["current_job_info"]["job_id"])  # type: ignore
            if job and isinstance(job, Job) and job.status == Status.STARTED:
                return job
        return None

    def heartbeat(self) -> None:
        current_job = self.get_current_job()
        if current_job:
            current_job.update(last_heartbeat=get_datetime())

    def get_zombies(self) -> List[Job]:
        started_jobs = Job.objects(status=Status.STARTED)
        max_missing_heartbeats = self.app_config.worker.max_missing_heartbeats
        heartbeat_time_interval_seconds = self.app_config.worker.heartbeat_time_interval_seconds
        if max_missing_heartbeats >= 1:
            return [
                job
                for job in started_jobs
                if (
                    job.last_heartbeat is not None
                    and get_datetime()
                    >= pytz.UTC.localize(job.last_heartbeat)
                    + timedelta(seconds=max_missing_heartbeats * heartbeat_time_interval_seconds)
                )
                or (
                    job.last_heartbeat is None
                    and job.started_at is not None
                    and get_datetime()
                    >= pytz.UTC.localize(job.started_at)
                    + timedelta(seconds=max_missing_heartbeats * heartbeat_time_interval_seconds)
                )
            ]
        else:
            return []

    def kill_zombies(self) -> None:
        zombies = self.get_zombies()
        if zombies:
            zombies_examples = [str(zombie.pk) for zombie in zombies[:10]]
            zombies_examples_str = ", ".join(zombies_examples) + (
                "..." if len(zombies_examples) != len(zombies) else ""
            )
            logging.info(f"Killing {len(zombies)} zombies. Job ids = " + zombies_examples_str)
            Job.objects(pk__in=[zombie.pk for zombie in zombies]).update(
                status=Status.ERROR, finished_at=get_datetime()
            )
            for zombie in zombies:
                job_info = JobInfo(
                    job_id=str(zombie.pk),
                    type=zombie.type,
                    dataset=zombie.dataset,
                    config=zombie.config,
                    split=zombie.split,
                    force=zombie.force,
                    priority=zombie.priority,
                )
                job_runner = self.job_runner_factory.create_job_runner(job_info)
                error = WorkerCrashedError("Worker crashed while running this job.")
                upsert_response(
                    kind=job_runner.processing_step.cache_kind,
                    dataset=job_runner.dataset,
                    config=job_runner.config,
                    split=job_runner.split,
                    content=dict(error.as_response()),
                    http_status=error.status_code,
                    error_code=error.code,
                    details=dict(error.as_response_with_cause()),
                    worker_version=job_runner.get_version(),
                    dataset_git_revision=job_runner.get_dataset_git_revision(),
                )
                logging.debug(
                    "response for"
                    f" dataset={job_runner.dataset} config={job_runner.config} split={job_runner.split} had an error,"
                    " cache updated"
                )

    def is_worker_alive(self, worker_loop_executor: OutputExecutor) -> bool:
        if not worker_loop_executor.running():
            worker_loop_executor.stop()  # raises an error if the worker returned exit code 1
            return False
        else:
            return True


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "WORKER_STATE_PATH" not in os.environ:
            os.environ["WORKER_STATE_PATH"] = os.path.join(tmp_dir, WORKER_STATE_FILE_NAME)

        app_config = AppConfig.from_env()
        init_logging(log_level=app_config.common.log_level)

        init_logging(log_level=app_config.common.log_level)
        # ^ set first to have logs as soon as possible
        assets_directory = init_assets_dir(directory=app_config.assets.storage_directory)

        processing_graph = ProcessingGraph(app_config.processing_graph.specification)

        with (
            LibrariesResource(
                hf_endpoint=app_config.common.hf_endpoint,
                init_hf_datasets_cache=app_config.datasets_based.hf_datasets_cache,
                numba_path=app_config.numba.path,
            ) as libraries_resource,
            CacheMongoResource(
                database=app_config.cache.mongo_database, host=app_config.cache.mongo_url
            ) as cache_resource,
            QueueMongoResource(
                database=app_config.queue.mongo_database, host=app_config.queue.mongo_url
            ) as queue_resource,
        ):
            if not cache_resource.is_available():
                raise RuntimeError("The connection to the cache database could not be established. Exiting.")
            if not queue_resource.is_available():
                raise RuntimeError("The connection to the queue database could not be established. Exiting.")

            job_runner_factory = JobRunnerFactory(
                app_config=app_config,
                processing_graph=processing_graph,
                hf_datasets_cache=libraries_resource.hf_datasets_cache,
                assets_directory=assets_directory,
            )
            worker_executor = WorkerExecutor(app_config, job_runner_factory)
            worker_executor.start()
