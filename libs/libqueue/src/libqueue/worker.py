# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from packaging import version
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from libqueue.config import QueueConfig
from libqueue.queue import EmptyQueueError, Queue


def parse_version(string_version: str) -> version.Version:
    parsed_version = version.parse(string_version)
    if isinstance(parsed_version, version.LegacyVersion):
        raise ValueError(f"LegacyVersion is not supported: {parsed_version}")
    return parsed_version


def compare_major_version(semver_a: str, semver_b: str) -> int:
    """
    Compare the major version of both semver arguments.

    Args:
        semver_a (:obj:`str`): the first semantic version
        semver_b (:obj:`str`): the second semantic version

    Returns:
        :obj:`int`: the difference between the major version of both semver arguments.
          0 if they are equal. Negative if semver_a is lower than semver_b, positive otherwise.
    Raises:
        :obj:`ValueError`: if a version passed as an argument is not a valid semantic version.
    """
    try:
        return parse_version(semver_a).major - parse_version(semver_b).major
    except Exception as err:
        raise RuntimeError(f"Could not get major versions: {err}") from err


class Worker(ABC):
    queue_config: QueueConfig
    version: str

    @property
    @abstractmethod
    def queue(self) -> Queue:
        pass

    def __init__(self, queue_config: QueueConfig, version: str) -> None:
        self.queue_config = queue_config
        self.version = version

    def has_memory(self) -> bool:
        if self.queue_config.max_memory_pct <= 0:
            return True
        virtual_memory_used: int = virtual_memory().used  # type: ignore
        virtual_memory_total: int = virtual_memory().total  # type: ignore
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.queue_config.max_memory_pct
        if not ok:
            logging.info(
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
            logging.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.queue_config.max_load_pct}%")
        return ok

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.queue_config.sleep_seconds * jitter
        logging.debug(f"sleep during {duration:.2f} seconds")
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
            logging.critical(f"quit due to an uncaught error while processing the job: {e}")
            raise

    def process_next_job(self) -> bool:
        logging.debug("try to process a job")

        try:
            job_id, dataset, config, split = self.queue.start_job()
            parameters_for_log = "dataset={dataset}" + ("" if split is None else f"config={config} split={split}")
            logging.debug(f"job assigned: {job_id} for {parameters_for_log}")
        except EmptyQueueError:
            logging.debug("no job in the queue")
            return False

        try:
            logging.info(f"compute {parameters_for_log}")
            success = self.compute(
                dataset=dataset,
                config=config,
                split=split,
            )
        finally:
            self.queue.finish_job(job_id=job_id, success=success)
            result = "success" if success else "error"
            logging.debug(f"job finished with {result}: {job_id} for {parameters_for_log}")
        return True

    def is_major_version_lower_than_worker(self, version: str) -> bool:
        """
        Checks if the version passed as an argument has a strictly lower major version than the worker's version.

        Args:
            version (:obj:`str`): the version to compare to the worker's version

        Returns:
            :obj:`bool`: True if the version passed as an argument has a strictly lower major version than the
              worker's version.
        Raises:
            :obj:`ValueError`: if a version passed as an argument is not a valid semantic version.
        """
        return compare_major_version(version, self.version) < 0

    @abstractmethod
    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        pass
