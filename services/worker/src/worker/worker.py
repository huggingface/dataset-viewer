# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from .constants import (
    DEFAULT_MAX_LOAD_PCT,
    DEFAULT_MAX_MEMORY_PCT,
    DEFAULT_WORKER_SLEEP_SECONDS,
)
from .utils import Queues

logger = logging.getLogger(__name__)


class Worker(ABC):
    queues: Queues
    max_load_pct: int
    max_memory_pct: int
    sleep_seconds: int

    def __init__(
        self,
        max_jobs_per_dataset: Optional[int] = None,
        max_load_pct: Optional[int] = None,
        max_memory_pct: Optional[int] = None,
        sleep_seconds: Optional[int] = None,
    ) -> None:
        self.queues = Queues(max_jobs_per_dataset=max_jobs_per_dataset)
        self.max_load_pct = DEFAULT_MAX_LOAD_PCT if max_load_pct is None else max_load_pct
        self.max_memory_pct = DEFAULT_MAX_MEMORY_PCT if max_memory_pct is None else max_memory_pct
        self.sleep_seconds = DEFAULT_WORKER_SLEEP_SECONDS if sleep_seconds is None else sleep_seconds

    def has_memory(self) -> bool:
        if self.max_memory_pct <= 0:
            return True
        virtual_memory_used: int = virtual_memory().used  # type: ignore
        virtual_memory_total: int = virtual_memory().total  # type: ignore
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.max_memory_pct
        if not ok:
            logger.info(f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {self.max_memory_pct}%")
        return ok

    def has_cpu(self) -> bool:
        if self.max_load_pct <= 0:
            return True
        load_pct = max(getloadavg()[:2]) / cpu_count() * 100
        # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
        ok = load_pct < self.max_load_pct
        if not ok:
            logger.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.max_load_pct}%")
        return ok

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.sleep_seconds * jitter
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

    @abstractmethod
    def process_next_job(self) -> bool:
        pass
