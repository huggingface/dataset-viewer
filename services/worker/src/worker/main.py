import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import timedelta
from typing import Any, Callable, List, Optional

import pytz
from filelock import FileLock
from libcommon.log import init_logging
from libcommon.queue import Job, Status, get_datetime
from libcommon.resources import QueueMongoResource
from mirakuru import OutputExecutor

from worker import start_worker_loop
from worker.config import AppConfig
from worker.loop import WorkerState

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


class WorkerExecutor:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config

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
            zombies_examples = [zombie.pk for zombie in zombies[:10]]
            zombies_examples_str = ", ".join(zombies_examples) + (
                "..." if len(zombies_examples) != len(zombies) else ""
            )
            logging.info(f"Killing {len(zombies)} zombies. Job ids = " + zombies_examples_str)
            Job.objects(pk__in=[zombie.pk for zombie in zombies]).update(
                status=Status.ERROR, finished_at=get_datetime()
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

        with QueueMongoResource(
            database=app_config.queue.mongo_database, host=app_config.queue.mongo_url
        ) as queue_resource:
            if not queue_resource.is_available():
                raise RuntimeError("The connection to the queue database could not be established. Exiting.")
            worker_executor = WorkerExecutor(app_config)
            worker_executor.start()
