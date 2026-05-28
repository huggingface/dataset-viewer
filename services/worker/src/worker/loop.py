# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import gc
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict

import orjson
from filelock import FileLock
from huggingface_hub import HfFileSystem
from libcommon.dtos import JobInfo
from libcommon.prometheus import LongStepProfiler, StepProfiler
from libcommon.queue.jobs import (
    AlreadyStartedJobError,
    EmptyQueueError,
    LockTimeoutError,
    NoWaitingJobError,
    Queue,
)
from libcommon.utils import get_datetime
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from worker.config import AppConfig
from worker.job_manager import JobManager
from worker.job_runner_factory import BaseJobRunnerFactory


class WorkerState(TypedDict):
    current_job_info: Optional[JobInfo]
    last_updated: datetime


class SigtermNotified(Exception):
    """a sigterm file used to notify a sigterm was sent by k8s to ask the loop to stop"""

    pass


@dataclass
class Loop:
    """
    A loop gets jobs from a queue and processes them.

    Once initialized, the loop can be started with the `run` method and will run until an uncaught exception
    is raised.

    The loop gracefully stops between jobs if a sigterm file is placed next to the state file.

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
        self.num_jobs_since_last_hffs_cache_clear = 0

    def has_memory(self) -> bool:
        """Deprecated: use has_pod_memory() + has_system_memory() instead.
        Kept for backward compatibility in other code that may call this.
        Returns True if EITHER pod-level OR system-level memory is OK."""
        return self.has_pod_memory() or self.has_system_memory()

    def _get_pod_memory_limit_bytes(self) -> Optional[int]:
        """Read the pod's memory limit from cgroup v1 or v2."""
        # Try cgroup v2 first (newer kernels)
        for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
            try:
                with open(path, "r") as f:
                    val = f.read().strip()
                    if val == "max":
                        return None  # No limit set
                    return int(val)
            except (FileNotFoundError, ValueError, PermissionError):
                continue
        return None

    def _get_pod_memory_usage_bytes(self) -> Optional[int]:
        """Read the pod's memory usage from cgroup v1 or v2."""
        # Try cgroup v2 first (newer kernels)
        for path in ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]:
            try:
                with open(path, "r") as f:
                    return int(f.read().strip())
            except (FileNotFoundError, ValueError, PermissionError):
                continue
        return None

    def has_pod_memory(self) -> bool:
        """Check if THIS pod is within its memory limit (prevents OOM-SIGKILL).

        Reads the pod's memory limit and usage directly from cgroup, which is accurate
        for containers regardless of how many other pods share the node.
        If the pod has no memory limit (cgroup returns None), falls back to system check.
        """
        if self.app_config.worker.max_memory_pct <= 0:
            return True

        pod_limit = self._get_pod_memory_limit_bytes()
        pod_usage = self._get_pod_memory_usage_bytes()

        if pod_limit and pod_usage:
            # Pod-level check: compare against the pod's cgroup limit
            percent = pod_usage / pod_limit
            threshold = self.app_config.worker.max_memory_pct / 100
        else:
            # Fallback to system memory if cgroup is not available
            total = swap_memory().total + virtual_memory().total
            used = swap_memory().used + virtual_memory().used
            percent = used / total if total > 0 else 0
            threshold = self.app_config.worker.max_memory_pct / 100

        ok = percent < threshold
        source = "cgroup" if (pod_limit and pod_usage) else "system"
        if not ok:
            logging.info(
                f"pod memory usage too high: {percent:.0%} vs {threshold:.0%} "
                f"(limit: {pod_limit // 1024**3 if pod_limit else 'N/A'}Gi, "
                f"used: {pod_usage // 1024**3 if pod_usage else 'N/A'}Gi, "
                f"source: {source})"
            )
        return ok

    def has_system_memory(self) -> bool:
        """Check if the NODE is within memory limits (prevents K8s eviction cascade).

        When the node is under memory pressure, K8s may SIGTERM all pods simultaneously
        to reclaim resources. This check prevents contributing to system-wide pressure.
        """
        if self.app_config.worker.max_system_memory_pct <= 0:
            return True

        total = swap_memory().total + virtual_memory().total
        used = swap_memory().used + virtual_memory().used
        percent = used / total if total > 0 else 0
        threshold = self.app_config.worker.max_system_memory_pct / 100

        ok = percent < threshold
        if not ok:
            logging.info(
                f"system memory usage too high: {percent:.0%} vs {threshold:.0%} "
                f"(limit: {total // 1024**3}Gi, used: {used // 1024**3}Gi)"
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
        # Pod-level check: prevents THIS pod from OOM-SIGKILL
        if not self.has_pod_memory():
            return False
        # System-level check: prevents system-wide pressure that triggers K8s eviction
        if not self.has_system_memory():
            return False
        return self.has_cpu()

    def sleep(self) -> None:
        # Free memory accumulated between jobs (Python reference cycles, freed objects)
        # This gives the GC a chance to reclaim memory before we re-check has_resources()
        gc.collect()
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.app_config.worker.sleep_seconds * jitter
        logging.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def check_sigterm_notification(self) -> None:
        if os.path.exists(os.path.join(os.path.dirname(self.state_file_path), "sigterm")):
            raise SigtermNotified()

    def run(self) -> None:
        logging.info("Worker loop started")
        try:
            while True:
                self.check_sigterm_notification()
                if self.has_resources() and self.process_next_job():
                    # loop immediately to try another job
                    # see https://github.com/huggingface/dataset-viewer/issues/265
                    continue
                self.check_sigterm_notification()
                with StepProfiler("loop", "sleep"):
                    self.sleep()
        except SigtermNotified:
            logging.exception("gracefully quit due to sigterm")
        except BaseException as err:
            logging.exception(f"quit due to an uncaught error: {err}")
            raise

    def process_next_job(self) -> bool:
        if self.num_jobs_since_last_hffs_cache_clear >= self.app_config.worker.num_jobs_between_hffs_cache_clear:
            HfFileSystem.clear_instance_cache()
            self.num_jobs_since_last_hffs_cache_clear = 0
        else:
            self.num_jobs_since_last_hffs_cache_clear += 1

        logging.debug("try to process a job")

        with StepProfiler("loop", "start_job"):
            try:
                job_info = self.queue.start_job(
                    difficulty_min=self.app_config.worker.difficulty_min,
                    difficulty_max=self.app_config.worker.difficulty_max,
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
            job_results = job_manager.run_job()
            for job_result in job_results:
                job_manager.save_job_result(job_result)

        with StepProfiler("loop", "finish_job"):
            job_manager.finish()
            self.set_worker_state(current_job_info=None)
            return True

    def set_worker_state(self, current_job_info: Optional[JobInfo]) -> None:
        worker_state: WorkerState = {"current_job_info": current_job_info, "last_updated": get_datetime()}
        with FileLock(f"{self.state_file_path}.lock"):
            with open(self.state_file_path, "wb") as worker_state_f:
                worker_state_f.write(orjson.dumps(worker_state))
