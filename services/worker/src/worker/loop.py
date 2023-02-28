# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from filelock import FileLock
from libcommon.queue import EmptyQueueError, JobInfo, Queue
from psutil import cpu_count, disk_usage, getloadavg, swap_memory, virtual_memory

from worker.config import WorkerConfig
from worker.job_runner_factory import BaseJobRunnerFactory


class UnknownJobTypeError(Exception):
    pass


class WorkerState(TypedDict):
    current_job_info: Optional[JobInfo]


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
        library_cache_paths (`set[str]`):
            The paths of the library caches. Used to check if the disk is full.
        worker_config (`WorkerConfig`):
            Worker configuration.
        max_jobs_per_namespace (`int`):
            The maximum number of jobs that can be processed per namespace. If a namespace has more jobs, the loop will
            wait until some jobs are finished.
    """

    job_runner_factory: BaseJobRunnerFactory
    library_cache_paths: set[str]
    worker_config: WorkerConfig
    max_jobs_per_namespace: int

    storage_paths: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.queue = Queue(max_jobs_per_namespace=self.max_jobs_per_namespace)
        self.storage_paths = set(self.worker_config.storage_paths).union(self.library_cache_paths)

    def has_memory(self) -> bool:
        if self.worker_config.max_memory_pct <= 0:
            return True
        virtual_memory_used = int(virtual_memory().used)
        virtual_memory_total = int(virtual_memory().total)
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.worker_config.max_memory_pct
        if not ok:
            logging.info(
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
            logging.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.worker_config.max_load_pct}%")
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
        logging.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def run(self) -> None:
        logging.info("Worker loop started")
        try:
            while True:
                if self.has_resources() and self.process_next_job():
                    # loop immediately to try another job
                    # see https://github.com/huggingface/datasets-server/issues/265
                    continue
                self.sleep()
        except BaseException:
            logging.exception("quit due to an uncaught error while processing the job")
            raise

    def process_next_job(self) -> bool:
        logging.debug("try to process a job")

        try:
            job_info = self.queue.start_job(only_job_types=self.worker_config.only_job_types)
            if self.worker_config.only_job_types and job_info["type"] not in self.worker_config.only_job_types:
                raise UnknownJobTypeError(
                    f"Job of type {job_info['type']} is not supported (only"
                    f" ${', '.join(self.worker_config.only_job_types)}). The queue should not have provided this"
                    " job. It is in an inconsistent state. Please report this issue to the datasets team."
                )
            self.set_worker_state(current_job_info=job_info)
            logging.debug(f"job assigned: {job_info}")
        except EmptyQueueError:
            self.set_worker_state(current_job_info=None)
            logging.debug("no job in the queue")
            return False

        job_runner = self.job_runner_factory.create_job_runner(job_info)
        finished_status = job_runner.run()
        self.queue.finish_job(job_id=job_runner.job_id, finished_status=finished_status)
        self.set_worker_state(current_job_info=None)
        logging.debug(f"job finished with {finished_status.value}: {job_runner}")
        return True

    def set_worker_state(self, current_job_info: Optional[JobInfo]) -> None:
        worker_state: WorkerState = {"current_job_info": current_job_info}
        if self.worker_config.state_file_path:
            with FileLock(f"{self.worker_config.state_file_path}.lock"):
                with open(self.worker_config.state_file_path, "w") as worker_state_f:
                    json.dump(worker_state, worker_state_f)
