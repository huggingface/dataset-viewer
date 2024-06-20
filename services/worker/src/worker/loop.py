# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict

import orjson
from filelock import FileLock
from libcommon.dtos import JobInfo
from libcommon.new_queue.jobs import (
    AlreadyStartedJobError,
    EmptyQueueError,
    LockTimeoutError,
    NoWaitingJobError,
    Queue,
)
from libcommon.prometheus import LongStepProfiler, StepProfiler
from libcommon.utils import get_datetime
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from worker.config import AppConfig
from worker.job_manager import JobManager
from worker.job_runner_factory import BaseJobRunnerFactory


class WorkerState(TypedDict):
    current_job_info: Optional[JobInfo]
    last_updated: datetime


@dataclass
class Loop:
    """
    A loop gets jobs from a queue and processes them.

    Once initialized, the loop can be started with the `run` method and will run until an uncaught exception
    is raised.

    Args:
        job_runner_factory (`JobRunnerFactory`):
            The job runner factory that will create a job runner for each job. Must be able to process the jobs of the
              queue.
        worker_config (`WorkerConfig`):
            Worker configuration.
        state_file_path (`str`):
            The path of the file where the state of the loop will be saved.
    """

    job_runner_factory: BaseJobRunnerFactory
    app_config: AppConfig
    state_file_path: str

    def __post_init__(self) -> None:
        self.queue = Queue()

    def has_memory(self) -> bool:
        if self.app_config.worker.max_memory_pct <= 0:
            return True
        virtual_memory_used = int(virtual_memory().used)
        virtual_memory_total = int(virtual_memory().total)
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.app_config.worker.max_memory_pct
        if not ok:
            logging.info(
                f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is"
                f" {self.app_config.worker.max_memory_pct}%"
            )
        return ok

    def has_cpu(self) -> bool:
        if self.app_config.worker.max_load_pct <= 0:
            return True
        load_pct = max(getloadavg()[:2]) / cpu_count() * 100
        # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
        ok = load_pct < self.app_config.worker.max_load_pct
        if not ok:
            logging.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.app_config.worker.max_load_pct}%")
        return ok

    def has_resources(self) -> bool:
        return self.has_memory() and self.has_cpu()

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.app_config.worker.sleep_seconds * jitter
        logging.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def run(self) -> None:
        logging.info("Worker loop started")
        try:
            while True:
                if self.has_resources() and self.process_next_job():
                    # loop immediately to try another job
                    # see https://github.com/huggingface/dataset-viewer/issues/265
                    continue
                with StepProfiler("loop", "sleep"):
                    self.sleep()
        except BaseException as err:
            logging.exception(f"quit due to an uncaught error: {err}")
            raise

    def process_next_job(self) -> bool:
        logging.debug("try to process a job")

        with StepProfiler("loop", "start_job"):
            try:
                job_info = self.queue.start_job(
                    difficulty_min=self.app_config.worker.difficulty_min,
                    difficulty_max=self.app_config.worker.difficulty_max,
                    job_types_blocked=self.app_config.worker.job_types_blocked,
                    job_types_only=self.app_config.worker.job_types_only,
                )
                self.set_worker_state(current_job_info=job_info)
                logging.debug(f"job assigned: {job_info}")
            except (EmptyQueueError, AlreadyStartedJobError, LockTimeoutError, NoWaitingJobError) as e:
                self.set_worker_state(current_job_info=None)
                logging.debug(e)
                return False

        with LongStepProfiler("loop", "run_job"):
            job_runner = self.job_runner_factory.create_job_runner(job_info)
            job_manager = JobManager(job_info=job_info, app_config=self.app_config, job_runner=job_runner)
            job_result = job_manager.run_job()

        with StepProfiler("loop", "finish_job"):
            job_manager.finish(job_result=job_result)
            self.set_worker_state(current_job_info=None)
            return True

    def set_worker_state(self, current_job_info: Optional[JobInfo]) -> None:
        worker_state: WorkerState = {"current_job_info": current_job_info, "last_updated": get_datetime()}
        with FileLock(f"{self.state_file_path}.lock"):
            with open(self.state_file_path, "wb") as worker_state_f:
                worker_state_f.write(orjson.dumps(worker_state))
