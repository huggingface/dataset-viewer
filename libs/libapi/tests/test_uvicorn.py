# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

MiB = 2**20
# a worker drains above a third of the pod limit, with 2 workers
WORKER_MEMORY_LIMIT = 300 * MiB

APP = """
import asyncio
import os

hoard = []


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            else:
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["path"] == "/grow":
        # filled with ones, so that the pages are really allocated
        hoard.append(bytearray(b"\\x01") * (400 * 2**20))
        # still in flight when the supervisor asks the worker to stop
        await asyncio.sleep(3)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(os.getpid()).encode()})
"""

SUPERVISOR = """
import sys
from pathlib import Path

from libapi import uvicorn

cgroup = Path(sys.argv[1])
uvicorn.CGROUP_MEMORY_MAX = cgroup / "memory.max"
uvicorn.CGROUP_MEMORY_STAT = cgroup / "memory.stat"
uvicorn.run("test_app:app", host="127.0.0.1", port=int(sys.argv[2]), workers=2, factory=False, replace_fat_workers=True)
"""


@dataclass
class Supervisor:
    cgroup: Path
    log: Path
    port: int

    def get(self, path: str = "/") -> int:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as response:
            return int(response.read())

    def set_pod_anonymous_memory(self, value: int) -> None:
        (self.cgroup / "memory.stat").write_text(f"anon {value}\nfile 0\n")

    def wait_for_log(self, text: str, timeout: float = 20) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in self.log.read_text().splitlines():
                if text in line:
                    return line
            time.sleep(0.1)
        raise AssertionError(f"{text!r} not in the supervisor logs:\n{self.log.read_text()}")


def is_running(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def wait_until_stopped(pid: int, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while is_running(pid):
        assert time.monotonic() < deadline, f"process {pid} still runs"
        time.sleep(0.1)


@pytest.fixture
def supervisor(tmp_path: Path) -> Iterator[Supervisor]:
    (tmp_path / "test_app.py").write_text(APP)
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text(f"{3 * WORKER_MEMORY_LIMIT}\n")
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    log = tmp_path / "supervisor.log"
    supervisor = Supervisor(cgroup=cgroup, log=log, port=port)
    supervisor.set_pod_anonymous_memory(0)
    with log.open("w") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-c", SUPERVISOR, str(cgroup), str(port)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        supervisor.wait_for_log("Application startup complete")
        yield supervisor
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_a_fat_worker_finishes_its_requests_and_is_replaced(supervisor: Supervisor) -> None:
    with ThreadPoolExecutor() as executor:
        fat_worker_response = executor.submit(supervisor.get, "/grow")
        drain = supervisor.wait_for_log("it finishes its requests while a new process takes its place")
        # the worker closes its listening socket at its next tick
        time.sleep(0.5)
        # the other worker and the new one serve the requests sent during the drain
        other_pids = {supervisor.get() for _ in range(10)}
        fat_worker = fat_worker_response.result()

    assert f"[{fat_worker}]" in drain
    assert fat_worker not in other_pids
    wait_until_stopped(fat_worker)
    assert supervisor.get() != fat_worker


def test_the_biggest_worker_is_killed_when_the_pod_gets_close_to_its_limit(supervisor: Supervisor) -> None:
    supervisor.set_pod_anonymous_memory(3 * WORKER_MEMORY_LIMIT)
    kill = supervisor.wait_for_log("Killing child process")
    supervisor.set_pod_anonymous_memory(0)
    killed = int(kill.split("[")[1].split("]")[0])

    wait_until_stopped(killed)
    supervisor.wait_for_log(f"Child process [{killed}] died")
    assert killed not in {supervisor.get() for _ in range(10)}
