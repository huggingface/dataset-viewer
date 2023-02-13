# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from dataclasses import dataclass, field

from libcommon.queue import EmptyQueueError, Queue
from psutil import cpu_count, disk_usage, getloadavg, swap_memory, virtual_memory

from worker.config import WorkerConfig
from worker.job_runner_factory import BaseJobRunnerFactory


@dataclass
class Loop:
    """
    A loop gets jobs from a queue and processes them.

    Once initialized, the loop can be started with the `run` method and will run until an uncaught exception
    is raised.

    Args:
        library_cache_paths (`set[str]`):
            The paths of the library caches. Used to check if the disk is full.
        queue (`Queue`):
            The job queue.
        job_runner_factory (`JobRunnerFactory`):
            The job runner factory that will create a job runner for each job. Must be able to process the jobs of the
              queue.
        worker_config (`WorkerConfig`):
            Worker configuration.
    """

    library_cache_paths: set[str]
    queue: Queue
    job_runner_factory: BaseJobRunnerFactory
    worker_config: WorkerConfig

    storage_paths: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.storage_paths = set(self.worker_config.storage_paths).union(self.library_cache_paths)

    def log(self, level: int, msg: str) -> None:
        logging.log(level=level, msg=f"[{self.queue.type}] {msg}")

    def debug(self, msg: str) -> None:
        self.log(level=logging.DEBUG, msg=msg)

    def info(self, msg: str) -> None:
        self.log(level=logging.INFO, msg=msg)

    def critical(self, msg: str) -> None:
        self.log(level=logging.CRITICAL, msg=msg)

    def exception(self, msg: str) -> None:
        self.log(level=logging.ERROR, msg=msg)

    def has_memory(self) -> bool:
        if self.worker_config.max_memory_pct <= 0:
            return True
        virtual_memory_used: int = virtual_memory().used  # type: ignore
        virtual_memory_total: int = virtual_memory().total  # type: ignore
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.worker_config.max_memory_pct
        if not ok:
            self.info(
                f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {self.worker_config.max_memory_pct}%"
            )
        return ok

    def has_cpu(self) -> bool:
        if self.worker_config.max_load_pct <= 0:
            return True
        load_pct = max(getloadavg()[:2]) / cpu_count() * 100
        # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
        ok = load_pct < self.worker_config.max_load_pct
        if not ok:
            self.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.worker_config.max_load_pct}%")
        return ok

    def has_storage(self) -> bool:
        if self.worker_config.max_disk_usage_pct <= 0:
            return True
        for path in self.storage_paths:
            try:
                usage = disk_usage(path)
                if usage.percent >= self.worker_config.max_disk_usage_pct:
                    return False
            except Exception:
                # if we can't get the disk usage, we let the process continue
                return True
        return True

    def has_resources(self) -> bool:
        return self.has_memory() and self.has_cpu() and self.has_storage()

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.worker_config.sleep_seconds * jitter
        self.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def run(self) -> None:
        self.info("Worker started")
        try:
            while True:
                if self.has_resources() and self.process_next_job():
                    # loop immediately to try another job
                    # see https://github.com/huggingface/datasets-server/issues/265
                    continue
                self.sleep()
        except BaseException as e:
            self.critical(f"quit due to an uncaught error while processing the job: {e}")
            raise

    def process_next_job(self) -> bool:
        self.debug("try to process a job")

        try:
            job_runner = self.job_runner_factory.create_job_runner(self.queue.start_job())
            self.debug(f"job assigned: {job_runner}")
        except EmptyQueueError:
            self.debug("no job in the queue")
            return False

        finished_status = job_runner.run()
        self.queue.finish_job(job_id=job_runner.job_id, finished_status=finished_status)
        self.debug(f"job finished with {finished_status.value}: {job_runner}")
        return True
