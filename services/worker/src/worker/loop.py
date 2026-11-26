# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import ctypes
import ctypes.util
import gc
import glob
import logging
import os
import random
import signal
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.connection import Connection, Pipe
from typing import Optional, TypedDict

import orjson
import pyarrow as pa
from filelock import FileLock
from huggingface_hub import HfFileSystem
from libcommon.dtos import JobInfo, JobResult
from libcommon.prometheus import START_JOB_NO_JOB_TOTAL, LongStepProfiler, StepProfiler
from libcommon.queue.jobs import (
    AlreadyStartedJobError,
    EmptyQueueError,
    LockTimeoutError,
    NoWaitingJobError,
    Queue,
)
from libcommon.storage_client import StorageClient
from libcommon.utils import get_datetime
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from worker.config import AppConfig
from worker.job_manager import JobManager
from worker.job_runner_factory import BaseJobRunnerFactory
from worker.job_runners._job_runner_with_cache import JobRunnerWithCache

# glibc keeps freed memory in its arenas until asked to return it to the OS; malloc_trim is glibc-only.
_MALLOC_TRIM = getattr(ctypes.CDLL(ctypes.util.find_library("c")), "malloc_trim", None)
# The kernel kills the pod without a trace when its cgroup reaches its limit: the watchdog logs the running job when the
# memory gets close, which is the only trace left of the job that caused the OOM.
MEMORY_WATCHDOG_THRESHOLD = 0.9
MEMORY_WATCHDOG_INTERVAL_SECONDS = 1.0
# the memory can stay above the threshold for a while: don't log more than once in this interval
MEMORY_WATCHDOG_LOG_INTERVAL_SECONDS = 10.0
# how often the loop checks the pod memory while a job runs in its child process
JOB_MEMORY_CHECK_INTERVAL_SECONDS = 0.25
# The child process runs the next jobs too, with its connections already open, until it holds more private memory than
# this share of the pod limit, or ran this many jobs: then the loop replaces it, and its memory goes back to the OS. Its
# jobs write to pages it shares with the loop (the imports of their job runners), and copy them: on the light workers,
# that's already 11% of the pod limit after a few jobs.
JOB_CHILD_MAX_PRIVATE_MEMORY_PCT = 25
JOB_CHILD_MAX_JOBS = 200


def free_memory() -> None:
    """Drop the caches a job can leave behind and return the freed memory to the OS.

    The HfFileSystem instance cache keeps the listing of every repo a job opened: one entry per file, which reaches
    1M+ objects for a repo with tens of thousands of files.
    """
    HfFileSystem.clear_instance_cache()
    gc.collect()
    pa.default_memory_pool().release_unused()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


def _remove_prometheus_files(pid: int) -> None:
    """Remove the metric files of a job's child process: else every job adds files that each scrape reads.

    The job's own metrics are lost with them: the loop process keeps measuring the job steps.
    """
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if directory:
        for path in glob.glob(os.path.join(directory, f"*_{pid}.db")):
            os.remove(path)


def _get_private_memory_bytes(pid: int) -> Optional[int]:
    """The memory a process wrote to since it was forked: the pages it shares with its parent don't count."""
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r") as f:
            return 1024 * sum(
                int(line.split()[1]) for line in f if line.startswith(("Private_Dirty:", "Private_Clean:"))
            )
    except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
        return None


@dataclass
class JobChild:
    """A child process of the loop, that runs the jobs the loop sends through its connection."""

    pid: int
    connection: Connection
    jobs: int = 0


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
        self.blocked_by_pod_memory_since: Optional[float] = None
        self.blocked_by_pod_memory_deadline = 0.0
        self.current_job_info: Optional[JobInfo] = None
        self.current_job_started_at = 0.0
        self.job_child: Optional[JobChild] = None
        # the children that are replaced exit on their own: the loop reaps them later, instead of waiting for them
        self.exiting_job_children: set[int] = set()

    def start_memory_watchdog(self) -> None:
        """Log the running job when the pod memory gets close to its limit."""
        if not self._get_pod_memory_limit_bytes():
            logging.info("no pod memory limit to watch, the worker memory watchdog is not started")
            return
        threading.Thread(target=self._watch_memory, name="worker-memory-watchdog", daemon=True).start()

    def _watch_memory(self) -> None:
        last_log = float("-inf")
        while True:
            time.sleep(MEMORY_WATCHDOG_INTERVAL_SECONDS)
            ratio = self.pod_memory_ratio()
            now = time.monotonic()
            if (
                ratio is not None
                and ratio >= MEMORY_WATCHDOG_THRESHOLD
                and now - last_log >= MEMORY_WATCHDOG_LOG_INTERVAL_SECONDS
            ):
                self.log_memory_close_to_limit(ratio)
                last_log = now

    def log_memory_close_to_limit(self, ratio: float) -> None:
        job_info = self.current_job_info
        if job_info is None:
            job = "no job"
        else:
            params = job_info["params"]
            job = (
                f"job {job_info['type']} for dataset={params['dataset']!r} config={params['config']!r}"
                f" split={params['split']!r} for {time.monotonic() - self.current_job_started_at:.0f}s"
            )
        logging.warning(f"worker memory at {ratio:.0%} of the pod limit, running {job}")

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

        percent = self.pod_memory_ratio()
        source = "cgroup"
        if percent is None:
            # Fallback to system memory if cgroup is not available
            total = swap_memory().total + virtual_memory().total
            used = swap_memory().used + virtual_memory().used
            percent = used / total if total > 0 else 0
            source = "system"
        threshold = self.app_config.worker.max_memory_pct / 100

        ok = percent < threshold
        if not ok:
            pod_limit = self._get_pod_memory_limit_bytes()
            pod_usage = self._get_pod_memory_usage_bytes()
            logging.info(
                f"pod memory usage too high: {percent:.0%} vs {threshold:.0%} "
                f"(limit: {pod_limit // 1024**3 if pod_limit else 'N/A'}Gi, "
                f"used: {pod_usage // 1024**3 if pod_usage else 'N/A'}Gi, "
                f"source: {source})"
            )
        return ok

    def pod_unreclaimable_memory_ratio(self) -> Optional[float]:
        """The pod's anonymous and shared memory as a share of its cgroup limit, or None if the cgroup v2 doesn't report
        them. Unlike the page cache, the kernel can't reclaim this memory: it's what makes the kernel kill the pod."""
        pod_limit = self._get_pod_memory_limit_bytes()
        try:
            with open("/sys/fs/cgroup/memory.stat", "r") as f:
                stats = dict(line.split() for line in f)
            unreclaimable = int(stats["anon"]) + int(stats["shmem"])
        except (FileNotFoundError, ValueError, KeyError, PermissionError):
            return None
        return unreclaimable / pod_limit if pod_limit else None

    def pod_memory_ratio(self) -> Optional[float]:
        """The pod's memory usage as a share of its cgroup limit, or None if the cgroup doesn't report both."""
        pod_limit = self._get_pod_memory_limit_bytes()
        pod_usage = self._get_pod_memory_usage_bytes()
        if not pod_limit or not pod_usage:
            return None
        return pod_usage / pod_limit

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

    def sleep(self) -> None:
        # Free memory accumulated between jobs (Python reference cycles, freed objects)
        # This gives the GC a chance to reclaim memory before we re-check the memory limits
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
        self.start_memory_watchdog()
        try:
            while True:
                self.check_sigterm_notification()
                if not self.has_pod_memory():
                    if self.should_exit_for_pod_memory():
                        # return cleanly: the executor stops and Kubernetes restarts the container with fresh memory
                        return
                else:
                    self.blocked_by_pod_memory_since = None
                    if self.has_system_memory() and self.has_cpu() and self.process_next_job():
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
        finally:
            self.stop_job_children()

    def should_exit_for_pod_memory(self) -> bool:
        """Free memory the first time the pod memory is too high, and give up if it stays too high.

        Without this, a worker above the limit would never reach the cache clear in process_next_job() and would
        stay blocked until the pod is replaced.
        """
        now = time.monotonic()
        if self.blocked_by_pod_memory_since is None:
            self.blocked_by_pod_memory_since = now
            # jitter so that workers blocked at the same time don't all restart together
            self.blocked_by_pod_memory_deadline = now + self.app_config.worker.max_seconds_blocked_by_pod_memory * (
                1 + random.random() / 4  # nosec
            )
            logging.info("pod memory usage too high: freeing caches and memory before checking again")
            with StepProfiler("loop", "free_memory"):
                free_memory()
            return False
        if now < self.blocked_by_pod_memory_deadline:
            return False
        blocked_seconds = now - self.blocked_by_pod_memory_since
        logging.warning(f"pod memory usage still too high after {blocked_seconds:.0f}s: exiting to restart the worker")
        return True

    def is_pod_memory_above(self, pct: int) -> bool:
        ratio = self.pod_memory_ratio()
        return ratio is not None and ratio >= pct / 100

    def process_next_job(self) -> bool:
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
                START_JOB_NO_JOB_TOTAL.labels(reason=type(e).__name__).inc()
                self.set_worker_state(current_job_info=None)
                logging.debug(e)
                return False

        with LongStepProfiler("loop", "run_job"):
            job_runner = self.job_runner_factory.create_job_runner(job_info)
            job_manager = JobManager(job_info=job_info, app_config=self.app_config, job_runner=job_runner)
            if self.app_config.worker.kill_job_memory_pct > 0:
                job_results = self.run_job_in_child_process(job_manager)
            else:
                job_results = list(job_manager.run_job())
            for job_result in job_results or []:
                job_manager.save_job_result(job_result)

        with StepProfiler("loop", "finish_job"):
            if job_results is not None:
                job_manager.finish()
            self.set_worker_state(current_job_info=None)

        # the memory of a job run in a child process goes back to the OS when the loop replaces the child
        if self.app_config.worker.kill_job_memory_pct <= 0 and self.is_pod_memory_above(
            self.app_config.worker.free_memory_after_job_pct
        ):
            with StepProfiler("loop", "free_memory"):
                free_memory()
        return True

    def run_job_in_child_process(self, job_manager: JobManager) -> Optional[list[JobResult]]:
        """Compute the job in the child process of the loop, and return its results.

        When the pod memory gets close to its limit, the loop kills the child before the kernel kills the whole pod.

        Returns:
            `list[JobResult]`, *optional*: the results of the job, or None if the child didn't compute them: the job has
              then been saved as an error and finished.
        """
        self.reap_job_children()
        child = self.job_child or self.start_job_child()
        self.job_child = None
        # the child runs the job in the cache directory of the loop's job runner: the loop removes it if the child dies
        job_runner = job_manager.job_runner
        job = (
            job_manager.job_info,
            job_runner.planned_cache_subdirectory if isinstance(job_runner, JobRunnerWithCache) else None,
        )
        try:
            child.connection.send(job)
        except OSError:
            # the idle child died since its last job
            self.retire_job_child(child)
            child = self.start_job_child()
            child.connection.send(job)
        child.jobs += 1
        kill_ratio = self.app_config.worker.kill_job_memory_pct / 100
        try:
            while True:
                if child.connection.poll(JOB_MEMORY_CHECK_INTERVAL_SECONDS):
                    job_results, job_manager.shortcut_jobs_by_key = child.connection.recv()
                    if self.can_run_more_jobs(child):
                        self.job_child = child
                    else:
                        self.retire_job_child(child)
                    return job_results  # type: ignore[no-any-return]
                ratio = self.pod_unreclaimable_memory_ratio()
                if ratio is not None and ratio >= kill_ratio:
                    os.kill(child.pid, signal.SIGKILL)
                    logging.warning(
                        f"killed the running job: the anonymous and shared memory of the pod reached {ratio:.0%} of"
                        " its limit"
                    )
                    break
        except (EOFError, OSError):
            # the child died: the kernel killed it, or it crashed
            pass
        child.connection.close()
        _, status = os.waitpid(child.pid, 0)
        _remove_prometheus_files(child.pid)
        exit_code = os.waitstatus_to_exitcode(status)
        # the child didn't run the post_compute hook, which removes the cache directory of the job
        job_manager.job_runner.post_compute()
        if exit_code == -signal.SIGKILL:
            # killed by the loop, or by the kernel between two checks. The retryable errors backfill retries the job,
            # and its next jobs go to a bigger worker
            job_manager.set_exceeded_maximum_memory()
        else:
            job_manager.set_stuck(
                message=f"Job runner crashed while running this job (exit code {exit_code}).", reason="crashed"
            )
        return None

    def start_job_child(self) -> JobChild:
        loop_connection, child_connection = Pipe()
        # The loop process must not run polars itself: a child forked after polars started its thread pool deadlocks.
        # The objects of the loop stay shared with the child: else the child's gc writes to them, and copies them.
        gc.freeze()
        pid = os.fork()
        if pid == 0:
            loop_connection.close()
            os._exit(self.run_job_child(child_connection))
        gc.unfreeze()
        child_connection.close()
        return JobChild(pid=pid, connection=loop_connection)

    def run_job_child(self, connection: Connection) -> int:
        """Run the jobs that the loop sends, until the loop closes the connection."""
        # pymongo resets its connections in the child, the warning is noise
        warnings.filterwarnings("ignore", message="MongoClient opened before fork")
        try:
            storage_client = getattr(self.job_runner_factory, "storage_client", None)
            if isinstance(storage_client, StorageClient):
                storage_client.reset_after_fork()
            while True:
                try:
                    job_info, cache_subdirectory = connection.recv()
                except EOFError:
                    return 0
                job_runner = self.job_runner_factory.create_job_runner(job_info)
                if isinstance(job_runner, JobRunnerWithCache) and cache_subdirectory is not None:
                    job_runner.planned_cache_subdirectory = cache_subdirectory
                job_manager = JobManager(job_info=job_info, app_config=self.app_config, job_runner=job_runner)
                connection.send((list(job_manager.run_job()), job_manager.shortcut_jobs_by_key))
        except BaseException:
            logging.exception("the child process of the jobs crashed")
            return 1

    def can_run_more_jobs(self, child: JobChild) -> bool:
        if child.jobs >= JOB_CHILD_MAX_JOBS:
            return False
        private_memory = _get_private_memory_bytes(child.pid)
        pod_limit = self._get_pod_memory_limit_bytes()
        if private_memory is None or not pod_limit:
            return False
        return private_memory < pod_limit * JOB_CHILD_MAX_PRIVATE_MEMORY_PCT / 100

    def retire_job_child(self, child: JobChild) -> None:
        """Let the child exit on its own: closing the connection ends its loop. reap_job_children() waits for it."""
        child.connection.close()
        self.exiting_job_children.add(child.pid)

    def reap_job_children(self) -> None:
        for pid in list(self.exiting_job_children):
            exited_pid, _ = os.waitpid(pid, os.WNOHANG)
            if exited_pid:
                self.exiting_job_children.discard(pid)
                _remove_prometheus_files(pid)

    def stop_job_children(self) -> None:
        if self.job_child is not None:
            self.retire_job_child(self.job_child)
            self.job_child = None
        for pid in self.exiting_job_children:
            os.waitpid(pid, 0)
            _remove_prometheus_files(pid)
        self.exiting_job_children.clear()

    def set_worker_state(self, current_job_info: Optional[JobInfo]) -> None:
        self.current_job_info = current_job_info
        self.current_job_started_at = time.monotonic()
        worker_state: WorkerState = {"current_job_info": current_job_info, "last_updated": get_datetime()}
        with FileLock(f"{self.state_file_path}.lock"):
            with open(self.state_file_path, "wb") as worker_state_f:
                worker_state_f.write(orjson.dumps(worker_state))
