# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import asyncio
import json
import logging
import os
import sys
from typing import Any, Callable, Optional

from filelock import FileLock
from libcommon.queue import Queue
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
        loop.create_task(every(self.heartbeat, seconds=self.app_config.worker.heartbeat_interval_seconds))
        loop.create_task(every(self.kill_zombies, seconds=self.app_config.worker.kill_zombies_interval_seconds))
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

    def heartbeat(self) -> None:
        worker_state = self.get_state()
        if worker_state["current_job_info"]:
            Queue().heartbeat(job_id=worker_state["current_job_info"]["job_id"])

    def kill_zombies(self) -> None:
        max_missing_heartbeats = self.app_config.worker.max_missing_heartbeats
        heartbeat_interval_seconds = self.app_config.worker.heartbeat_interval_seconds
        max_seconds_without_heartbeat = heartbeat_interval_seconds * max_missing_heartbeats
        queue = Queue()
        zombies = queue.get_zombies(max_seconds_without_heartbeat=max_seconds_without_heartbeat)
        queue.kill_zombies(zombies)
        message = "Job runner crashed while running this job (missing heartbeats)."
        for zombie in zombies:
            job_runner = self.job_runner_factory.create_job_runner(zombie)
            job_runner.set_crashed(message=message)

    def is_worker_alive(self, worker_loop_executor: OutputExecutor) -> bool:
        if not worker_loop_executor.running():
            worker_loop_executor.stop()  # raises an error if the worker returned exit code 1
            return False
        else:
            return True
