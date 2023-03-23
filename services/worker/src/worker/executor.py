# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import orjson
from filelock import FileLock
from libcommon.queue import Queue, get_datetime
from mirakuru import OutputExecutor

from worker import start_worker_loop
from worker.config import AppConfig
from worker.job_runner_factory import JobRunnerFactory
from worker.loop import WorkerState

START_WORKER_LOOP_PATH = start_worker_loop.__file__


async def every(
    func: Callable[..., Optional[Any]], *args: Any, seconds: int, stop_on: Optional[Any] = None, **kwargs: Any
) -> None:
    while True:
        out = func(*args, **kwargs)
        if stop_on is not None and out == stop_on:
            break
        await asyncio.sleep(seconds)


class BadWorkerState(RuntimeError):
    """Raised when the worker state from the worker read by the executor is not valid."""

    pass


class WorkerExecutor:
    def __init__(self, app_config: AppConfig, job_runner_factory: JobRunnerFactory, state_file_path: str) -> None:
        self.app_config = app_config
        self.job_runner_factory = job_runner_factory
        self.state_file_path = state_file_path

        max_missing_heartbeats = self.app_config.worker.max_missing_heartbeats
        heartbeat_interval_seconds = self.app_config.worker.heartbeat_interval_seconds
        self.max_seconds_without_heartbeat_for_zombies = heartbeat_interval_seconds * max_missing_heartbeats

    def _create_worker_loop_executor(self) -> OutputExecutor:
        banner = self.state_file_path
        start_worker_loop_command = [
            sys.executable,
            START_WORKER_LOOP_PATH,
            "--print-worker-state-path",
        ]
        return OutputExecutor(start_worker_loop_command, banner, timeout=10)

    def start(self) -> None:
        exceptions = []
        worker_loop_executor = self._create_worker_loop_executor()
        worker_loop_executor.start()  # blocking until the banner is printed

        def custom_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            nonlocal exceptions
            # first, handle with default handler
            loop.default_exception_handler(context)

            exception = context.get("exception")
            if exception:
                exceptions.append(repr(exception))
                loop.stop()

        loop = asyncio.get_event_loop()
        loop.set_exception_handler(custom_exception_handler)
        logging.info("Starting heartbeat.")
        loop.create_task(every(self.heartbeat, seconds=self.app_config.worker.heartbeat_interval_seconds))
        loop.create_task(every(self.kill_zombies, seconds=self.app_config.worker.kill_zombies_interval_seconds))
        loop.create_task(
            every(
                self.kill_long_job,
                worker_loop_executor=worker_loop_executor,
                seconds=self.app_config.worker.kill_long_job_interval_seconds,
            )
        )
        loop.run_until_complete(
            every(self.is_worker_alive, worker_loop_executor=worker_loop_executor, seconds=1, stop_on=False)
        )
        if exceptions:
            raise RuntimeError(f"Some async tasks failed: {exceptions}")

    def get_state(self) -> Optional[WorkerState]:
        worker_state_file_path = self.state_file_path
        if not os.path.exists(worker_state_file_path):
            return None
        with FileLock(f"{worker_state_file_path}.lock"):
            try:
                with open(worker_state_file_path, "rb") as worker_state_f:
                    worker_state = orjson.loads(worker_state_f.read())
                    return WorkerState(
                        current_job_info=worker_state.get("current_job_info"),
                        last_updated=datetime.fromisoformat(worker_state["last_updated"]),
                    )
            except (orjson.JSONDecodeError, KeyError) as err:
                raise BadWorkerState(f"Failed to read worker state at {worker_state_file_path}") from err

    def heartbeat(self) -> None:
        worker_state = self.get_state()
        if worker_state and worker_state["current_job_info"]:
            Queue().heartbeat(job_id=worker_state["current_job_info"]["job_id"])

    def kill_zombies(self) -> None:
        queue = Queue()
        zombies = queue.get_zombies(max_seconds_without_heartbeat=self.max_seconds_without_heartbeat_for_zombies)
        queue.kill_zombies(zombies)
        message = "Job runner crashed while running this job (missing heartbeats)."
        for zombie in zombies:
            job_runner = self.job_runner_factory.create_job_runner(zombie)
            job_runner.set_crashed(message=message)

    def kill_long_job(self, worker_loop_executor: OutputExecutor) -> None:
        worker_state = self.get_state()
        if worker_state and worker_state["current_job_info"]:
            long_job = worker_state["current_job_info"]
            last_updated = worker_state["last_updated"]
            if last_updated + timedelta(seconds=self.app_config.worker.max_job_duration_seconds) <= get_datetime():
                _duration_seconds = int((get_datetime() - last_updated).total_seconds())
                logging.warning(
                    f"Job {long_job} exceeded maximum duration of"
                    f" {self.app_config.worker.max_job_duration_seconds} seconds ({_duration_seconds} seconds)."
                )
                try:
                    worker_loop_executor.stop()  # raises an error if the worker returned exit code 1
                finally:
                    Queue().kill_long_job(long_job)
                    job_runner = self.job_runner_factory.create_job_runner(long_job)
                    message = "Job runner was killed while running this job (job exceeded maximum duration)."
                    job_runner.set_exceeded_maximum_duration(message=message)

    def is_worker_alive(self, worker_loop_executor: OutputExecutor) -> bool:
        if not worker_loop_executor.running():
            worker_loop_executor.stop()  # raises an error if the worker returned exit code 1
            return False
        else:
            return True
