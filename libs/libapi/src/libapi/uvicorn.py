# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

"""Starting uvicorn so that respawned workers still have their secrets.

uvicorn starts each worker with the "spawn" method, so a worker is a fresh interpreter that re-imports
the app and rebuilds its config from scratch. Its supervisor also restarts a worker that dies, without
touching the container. Put together, a worker that dies once the read window has closed comes back to
blank files and crash-loops, while the container stays up and the window never reopens.

The parent read the secrets while the window was open, so it carries them to every worker it starts,
including the ones it starts hours later. They travel in the Config object uvicorn already pickles to
each worker, never through the environment or the disk.
"""

import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional, cast

import uvicorn
from libcommon.secrets import get_secrets, inherit_secrets
from prometheus_client import Counter
from uvicorn.supervisors import Multiprocess
from uvicorn.supervisors.multiprocess import Process

CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
CGROUP_MEMORY_STAT = Path("/sys/fs/cgroup/memory.stat")
# The kernel kills the whole pod when it cannot reclaim memory under the cgroup limit. The page cache is reclaimable,
# so only the anonymous memory counts.
POD_MEMORY_KILL_RATIO = 0.9
# a burst of requests fills a worker with GiBs in less than a second
POD_MEMORY_CHECK_SECONDS = 0.1
DRAIN_TIMEOUT_SECONDS = 60.0

WORKER_MEMORY_REPLACEMENTS = Counter(
    "uvicorn_worker_memory_replacements_total",
    "Number of uvicorn workers replaced because they held too much memory",
    ["action"],
)

logger = logging.getLogger("uvicorn.error")


class _Config(uvicorn.Config):
    """A uvicorn Config that carries the parent's secrets to the worker it is pickled into."""

    def __init__(self, *args: Any, secrets: Mapping[str, str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.secrets = secrets

    def load(self) -> None:
        # Runs in the worker, before the app factory builds any config from the environment.
        inherit_secrets(self.secrets)
        super().load()


def _read_cgroup_memory_max() -> Optional[int]:
    try:
        value = CGROUP_MEMORY_MAX.read_text().strip()
    except OSError:
        return None
    # memory.max is "max" when the cgroup has no limit
    return int(value) if value.isdigit() else None


def _read_pod_anonymous_memory() -> Optional[int]:
    try:
        for line in CGROUP_MEMORY_STAT.read_text().splitlines():
            key, value = line.split()
            if key == "anon":
                return int(value)
    except (OSError, ValueError):
        pass
    return None


def _read_worker_anonymous_memory(process: Process) -> int:
    try:
        for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


class _MemorySupervisor(Multiprocess):
    """A uvicorn supervisor that replaces the workers that hold too much memory, before the pod reaches its limit.

    A worker above its share of the pod limit drains: it stops taking new connections, finishes its requests and
    exits, while a new worker takes its place at once. Only one worker drains at a time. When the pod gets close to
    its limit anyway, the biggest worker is killed: its requests fail, but the kernel would kill the whole pod.
    The supervisor loop drains the workers, and a thread of its own kills them: the loop checks the workers every
    0.5s, and waits for each of them to answer a ping.
    """

    def __init__(
        self,
        config: uvicorn.Config,
        target: Callable[[Optional[list[socket.socket]]], None],
        sockets: list[socket.socket],
        memory_limit: int,
    ) -> None:
        super().__init__(config, target, sockets)
        self.memory_limit = memory_limit
        # the supervisor and the new worker of a drain need memory too
        self.worker_memory_limit = memory_limit // (config.workers + 1)
        self.draining: Optional[Process] = None
        self.drain_deadline = 0.0

    def run(self) -> None:
        threading.Thread(target=self.kill_the_biggest_worker_when_the_pod_is_full, daemon=True).start()
        super().run()

    def keep_subprocess_alive(self) -> None:
        if not self.should_exit.is_set():
            self.drain_fat_workers()
        super().keep_subprocess_alive()

    def join_all(self) -> None:
        super().join_all()
        if self.draining is not None:
            self.draining.join()

    def kill_the_biggest_worker_when_the_pod_is_full(self) -> None:
        while not self.should_exit.wait(POD_MEMORY_CHECK_SECONDS):
            pod_memory = _read_pod_anonymous_memory()
            if pod_memory is None or pod_memory < POD_MEMORY_KILL_RATIO * self.memory_limit:
                continue
            draining = self.draining
            workers = self.processes if draining is None else [*self.processes, draining]
            memory = {process: _read_worker_anonymous_memory(process) for process in workers}
            biggest = max(workers, key=memory.__getitem__)
            logger.warning(
                f"Killing child process [{biggest.pid}], which holds {memory[biggest] // 2**20} MiB: the pod holds"
                f" {pod_memory // 2**20} MiB of anonymous memory, for a limit of {self.memory_limit // 2**20} MiB"
            )
            biggest.kill()
            WORKER_MEMORY_REPLACEMENTS.labels(action="killed").inc()
            # the supervisor loop reaps it and starts a new worker: wait until its memory is freed
            deadline = time.monotonic() + 1
            while _read_worker_anonymous_memory(biggest) and time.monotonic() < deadline:
                time.sleep(0.01)

    def drain_fat_workers(self) -> None:
        if self.draining is not None:
            if self.draining.process.exitcode is None and time.monotonic() >= self.drain_deadline:
                logger.warning(
                    f"Child process [{self.draining.pid}] did not finish its requests in {DRAIN_TIMEOUT_SECONDS:.0f}s,"
                    " killing it"
                )
                self.draining.kill()
                self.draining.join()
            if self.draining.process.exitcode is not None:
                self.draining.join()
                self.draining = None

        if self.draining is None:
            memory = {process: _read_worker_anonymous_memory(process) for process in self.processes}
            biggest = max(self.processes, key=memory.__getitem__)
            if memory[biggest] >= self.worker_memory_limit:
                self.drain(biggest, memory[biggest])

    def drain(self, process: Process, memory: int) -> None:
        logger.warning(
            f"Child process [{process.pid}] holds {memory // 2**20} MiB, more than"
            f" {self.worker_memory_limit // 2**20} MiB: it finishes its requests while a new process takes its place"
        )
        # uvicorn's own terminate() also closes the pipe the worker answers the pings on, which crashes the thread
        # that answers them in the worker
        os.kill(cast(int, process.pid), signal.SIGTERM)
        new_process = Process(self.config, self.target, self.sockets)
        new_process.start()
        self.processes[self.processes.index(process)] = new_process
        self.draining = process
        self.drain_deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
        WORKER_MEMORY_REPLACEMENTS.labels(action="drained").inc()


def run(
    app: str, *, host: str, port: int, workers: int, factory: bool = True, replace_fat_workers: bool = False
) -> None:
    """Serve `app`, mirroring uvicorn.run for the cases we use.

    With `replace_fat_workers`, the workers that hold too much memory are replaced before the pod reaches its
    memory limit (see _MemorySupervisor).
    """
    config = _Config(app, host=host, port=port, workers=workers, factory=factory, secrets=dict(get_secrets()))
    server = uvicorn.Server(config)
    if workers > 1:
        sockets = [config.bind_socket()]
        memory_limit = _read_cgroup_memory_max() if replace_fat_workers else None
        if memory_limit is None:
            Multiprocess(config, target=server.run, sockets=sockets).run()
        else:
            _MemorySupervisor(config, target=server.run, sockets=sockets, memory_limit=memory_limit).run()
    else:
        server.run()
