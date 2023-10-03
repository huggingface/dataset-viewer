# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from random import random
from typing import Any, Optional, Union

import orjson
from filelock import FileLock
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.utils import get_datetime
from mirakuru import OutputExecutor, ProcessExitedWithError

from worker import start_worker_loop
from worker.config import AppConfig
from worker.job_manager import JobManager
from worker.job_runner_factory import JobRunnerFactory
from worker.loop import WorkerState

START_WORKER_LOOP_PATH = start_worker_loop.__file__


async def every(
    func: Callable[..., Optional[Any]],
    *args: Any,
    seconds: Union[float, tuple[float, float]],
    stop_on: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    while True:
        out = func(*args, **kwargs)
        if stop_on is not None and out == stop_on:
            break
        delay = (
            seconds[0] + (seconds[1] - seconds[0]) * random() if isinstance(seconds, tuple) else seconds  # nosec B311
        )
        await asyncio.sleep(delay)


class BadWorkerState(RuntimeError):
    """Raised when the worker state from the worker read by the executor is not valid."""

    pass


class WorkerExecutor:
    def __init__(self, app_config: AppConfig, job_runner_factory: JobRunnerFactory, state_file_path: str) -> None:
        self.app_config = app_config
        self.job_runner_factory = job_runner_factory
        self.state_file_path = state_file_path
        self.processing_graph = ProcessingGraph(self.app_config.processing_graph)

        max_missing_heartbeats = self.app_config.worker.max_missing_heartbeats
        heartbeat_interval_seconds = self.app_config.worker.heartbeat_interval_seconds
        self.max_seconds_without_heartbeat_for_zombies = heartbeat_interval_seconds * max_missing_heartbeats

        self.heartbeat_interval_seconds = self.app_config.worker.heartbeat_interval_seconds
        self.max_job_duration_seconds = self.app_config.worker.max_job_duration_seconds
        self.kill_zombies_interval_seconds = self.app_config.worker.kill_zombies_interval_seconds
        self.kill_long_job_interval_seconds = self.app_config.worker.kill_long_job_interval_seconds

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
        loop.create_task(every(self.heartbeat, seconds=self.heartbeat_interval_seconds))
        loop.create_task(
            every(
                self.kill_zombies,
                seconds=(
                    self.kill_zombies_interval_seconds * 0.5,
                    self.kill_zombies_interval_seconds * 1.5,
                ),
            )
        )
        loop.create_task(
            every(
                self.kill_long_job,
                worker_loop_executor=worker_loop_executor,
                seconds=(
                    self.kill_long_job_interval_seconds * 0.5,
                    self.kill_long_job_interval_seconds * 1.5,
                ),
            )
        )
        loop.run_until_complete(
            every(self.is_worker_alive, worker_loop_executor=worker_loop_executor, seconds=1.0, stop_on=False)
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
        message = "Job manager crashed while running this job (missing heartbeats)."
        for zombie in zombies:
            job_runner = self.job_runner_factory.create_job_runner(zombie)
            job_manager = JobManager(
                job_info=zombie,
                app_config=self.app_config,
                job_runner=job_runner,
                processing_graph=self.processing_graph,
            )
            job_manager.set_crashed(message=message)
            logging.info(f"Killing zombie. Job info = {zombie}")

    def kill_long_job(self, worker_loop_executor: OutputExecutor) -> None:
        worker_state = self.get_state()
        if worker_state and worker_state["current_job_info"]:
            long_job = worker_state["current_job_info"]
            last_updated = worker_state["last_updated"]
            coefficient = 10 if long_job["params"]["dataset"] == "cerebras/SlimPajama-627B" else 1
            if last_updated + timedelta(seconds=coefficient * self.max_job_duration_seconds) <= get_datetime():
                _duration_seconds = int((get_datetime() - last_updated).total_seconds())
                logging.warning(
                    f"Job {long_job} exceeded maximum duration of"
                    f" {self.max_job_duration_seconds} seconds ({_duration_seconds} seconds)."
                )
                try:
                    worker_loop_executor.stop()  # raises an error if the worker returned exit code 1
                finally:
                    logging.info(f"Killing a long job. Job info = {long_job}")
                    job_runner = self.job_runner_factory.create_job_runner(long_job)
                    job_manager = JobManager(
                        job_info=long_job,
                        app_config=self.app_config,
                        job_runner=job_runner,
                        processing_graph=self.processing_graph,
                    )
                    message = "Job manager was killed while running this job (job exceeded maximum duration)."
                    job_manager.set_exceeded_maximum_duration(message=message)

    def is_worker_alive(self, worker_loop_executor: OutputExecutor) -> bool:
        if worker_loop_executor.running():
            return True
        try:
            worker_loop_executor.stop()  # raises an error if the worker returned unexpected exit code
        except ProcessExitedWithError as err:
            explanation = f"exit code f{err.exit_code}"
            if err.exit_code == -9:
                explanation += " SIGKILL - surely an OOM"
            error_msg = f"Worker crashed ({explanation})"
            state = self.get_state()
            if state and state["current_job_info"]:
                error_msg += f"when running job_id={state['current_job_info']['job_id']}"
            logging.error(error_msg)
            raise
        return False
