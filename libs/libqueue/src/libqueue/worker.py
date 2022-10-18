# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from libqueue.config import QueueConfig
from libqueue.queue import EmptyQueue, Queue

logger = logging.getLogger(__name__)


class Worker(ABC):
    queue_config: QueueConfig

    @property
    @abstractmethod
    def queue(self) -> Queue:
        pass

    def __init__(self, queue_config: QueueConfig) -> None:
        self.queue_config = queue_config

    def has_memory(self) -> bool:
        if self.queue_config.max_memory_pct <= 0:
            return True
        virtual_memory_used: int = virtual_memory().used  # type: ignore
        virtual_memory_total: int = virtual_memory().total  # type: ignore
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.queue_config.max_memory_pct
        if not ok:
            logger.info(
                f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {self.queue_config.max_memory_pct}%"
            )
        return ok

    def has_cpu(self) -> bool:
        if self.queue_config.max_load_pct <= 0:
            return True
        load_pct = max(getloadavg()[:2]) / cpu_count() * 100
        # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
        ok = load_pct < self.queue_config.max_load_pct
        if not ok:
            logger.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.queue_config.max_load_pct}%")
        return ok

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.queue_config.sleep_seconds * jitter
        logger.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def loop(self) -> None:
        try:
            while True:
                if self.has_memory() and self.has_cpu() and self.process_next_job():
                    # loop immediately to try another job
                    # see https://github.com/huggingface/datasets-server/issues/265
                    continue
                self.sleep()
        except BaseException as e:
            logger.critical(f"quit due to an uncaught error while processing the job: {e}")
            raise

    def process_next_job(self) -> bool:
        logger.debug("try to process a job")

        try:
            job_id, dataset, config, split = self.queue.start_job()
            parameters_for_log = "dataset={dataset}" + ("" if split is None else f"config={config} split={split}")
            logger.debug(f"job assigned: {job_id} for {parameters_for_log}")
        except EmptyQueue:
            logger.debug("no job in the queue")
            return False

        try:
            logger.info(f"compute {parameters_for_log}")
            success = self.compute(
                dataset=dataset,
                config=config,
                split=split,
            )
        finally:
            self.queue.finish_job(job_id=job_id, success=success)
            result = "success" if success else "error"
            logger.debug(f"job finished with {result}: {job_id} for {parameters_for_log}")
        return True

    @abstractmethod
    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        pass
