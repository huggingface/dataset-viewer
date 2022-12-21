# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Type

from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from libcommon.config import CommonConfig, QueueConfig, WorkerLoopConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import EmptyQueueError, Queue
from libcommon.worker import Worker


@dataclass
class WorkerLoop:
    """
    A worker loop gets jobs from a queue and processes them.

    Once initialized, the worker loop can be started with the `loop` method and will run until an uncaught exception
    is raised.

    Args:
        common_config (`CommonConfig`):
            Common configuration.
        processing_step (`ProcessingStep`):
            The processing step supported by the worker loop (a worker loop can only process one step).
        queue_config (`QueueConfig`):
            Queue configuration.
        worker_class (`Type[Worker]`):
            The worker class to use to process the jobs. It must be compatible with the processing step.
        worker_loop_config (`WorkerLoopConfig`):
            Worker loop configuration.
    """

    common_config: CommonConfig
    processing_step: ProcessingStep
    queue_config: QueueConfig
    worker_class: Type[Worker]
    worker_loop_config: WorkerLoopConfig
    queue: Queue = field(init=False)

    def __post_init__(self) -> None:
        worker_endpoint = self.worker_class.get_endpoint()
        if self.processing_step.endpoint != worker_endpoint:
            raise ValueError(
                f"The processing step is {self.processing_step.endpoint}, but the worker processes {worker_endpoint}"
            )
        self.queue = Queue(
            type=self.processing_step.job_type, max_jobs_per_namespace=self.queue_config.max_jobs_per_namespace
        )

    def log(self, level: int, msg: str) -> None:
        logging.log(level=level, msg=f"[{self.processing_step.endpoint}] {msg}")

    def debug(self, msg: str) -> None:
        self.log(level=logging.DEBUG, msg=msg)

    def info(self, msg: str) -> None:
        self.log(level=logging.INFO, msg=msg)

    def critical(self, msg: str) -> None:
        self.log(level=logging.CRITICAL, msg=msg)

    def exception(self, msg: str) -> None:
        self.log(level=logging.ERROR, msg=msg)

    def has_memory(self) -> bool:
        if self.worker_loop_config.max_memory_pct <= 0:
            return True
        virtual_memory_used: int = virtual_memory().used  # type: ignore
        virtual_memory_total: int = virtual_memory().total  # type: ignore
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.worker_loop_config.max_memory_pct
        if not ok:
            self.info(
                f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is"
                f" {self.worker_loop_config.max_memory_pct}%"
            )
        return ok

    def has_cpu(self) -> bool:
        if self.worker_loop_config.max_load_pct <= 0:
            return True
        load_pct = max(getloadavg()[:2]) / cpu_count() * 100
        # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
        ok = load_pct < self.worker_loop_config.max_load_pct
        if not ok:
            self.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.worker_loop_config.max_load_pct}%")
        return ok

    def has_storage(self) -> bool:
        # placeholder, to be overridden if needed
        return True

    def has_resources(self) -> bool:
        return self.has_memory() and self.has_cpu() and self.has_storage()

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.worker_loop_config.sleep_seconds * jitter
        self.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def loop(self) -> None:
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
            worker = self.worker_class(
                self.queue.start_job(), common_config=self.common_config, processing_step=self.processing_step
            )
            self.debug(f"job assigned: {worker}")
        except EmptyQueueError:
            self.debug("no job in the queue")
            return False

        finished_status = worker.run()
        self.queue.finish_job(job_id=worker.job_id, finished_status=finished_status)
        self.debug(f"job finished with {finished_status.value}: {worker}")
        return True
