# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import ctypes
import ctypes.util
import gc
import itertools
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
from prometheus_client import Histogram

# the kernel kills the pod without any warning when its cgroup reaches memory.max: the watchdog logs what the workers
# are doing when the memory gets close to it, which is the only trace left of the request that caused the OOM
CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
PROC_SELF_STATUS = Path("/proc/self/status")
# glibc keeps the memory a request freed, so a worker can grow request after request: log the requests after which it
# holds this much more anonymous memory. The other requests served at the same time by the worker count too.
MEMORY_GROWTH_LOG_BYTES = 256 * 2**20
# glibc keeps freed memory in its arenas until asked to return it to the OS; malloc_trim is glibc-only.
_MALLOC_TRIM = getattr(ctypes.CDLL(ctypes.util.find_library("c")), "malloc_trim", None)
WATCHDOG_INTERVAL_SECONDS = 1.0
WATCHDOG_THRESHOLD = 0.85
# don't log the in-flight requests more than once in this interval, the memory can stay above the threshold
WATCHDOG_LOG_INTERVAL_SECONDS = 10.0

ROWS_ARROW_BYTES = Histogram(
    "rows_arrow_bytes",
    "Size in memory of the Arrow table read by a /rows request",
    buckets=[2**n for n in range(20, 34)],  # 1 MiB to 8 GiB
)


@dataclass
class InFlightRequest:
    description: str
    started_at: float = field(default_factory=time.monotonic)
    arrow_bytes: Optional[int] = None


_in_flight: dict[int, InFlightRequest] = {}
_in_flight_lock = threading.Lock()
_ids = itertools.count()
# set after a request that grew the worker: the watchdog thread then gives the freed memory back to the OS, once the
# request has returned and its tables and rows are no longer referenced
_release_requested = threading.Event()


def _read_anonymous_memory() -> Optional[int]:
    try:
        for line in PROC_SELF_STATUS.read_text().splitlines():
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


@contextmanager
def track_request(description: str) -> Iterator[InFlightRequest]:
    request = InFlightRequest(description=description)
    request_id = next(_ids)
    memory_before = _read_anonymous_memory()
    with _in_flight_lock:
        _in_flight[request_id] = request
    try:
        yield request
    finally:
        with _in_flight_lock:
            del _in_flight[request_id]
            other_requests = len(_in_flight)
        memory_after = _read_anonymous_memory()
        if memory_before is not None and memory_after is not None:
            growth = memory_after - memory_before
            if growth >= MEMORY_GROWTH_LOG_BYTES:
                logging.warning(
                    f"rows worker {os.getpid()} holds {growth // 2**20} MiB more anonymous memory"
                    f" ({memory_after // 2**20} MiB) after {request.description} for"
                    f" {time.monotonic() - request.started_at:.1f}s, arrow table of"
                    f" {'unknown' if request.arrow_bytes is None else request.arrow_bytes} bytes,"
                    f" with {other_requests} other requests in flight"
                )
                _release_requested.set()


def _release_freed_memory() -> None:
    memory_before = _read_anonymous_memory()
    gc.collect()
    pa.default_memory_pool().release_unused()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)
    memory_after = _read_anonymous_memory()
    if memory_before is not None and memory_after is not None:
        logging.info(
            f"rows worker {os.getpid()} gave {(memory_before - memory_after) // 2**20} MiB back to the OS, it holds"
            f" {memory_after // 2**20} MiB of anonymous memory"
        )


def record_arrow_bytes(request: InFlightRequest, arrow_bytes: int) -> None:
    request.arrow_bytes = arrow_bytes
    ROWS_ARROW_BYTES.observe(arrow_bytes)
    logging.info(f"/rows arrow table of {arrow_bytes} bytes for {request.description}")


def _read_int(path: Path) -> Optional[int]:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    # memory.max is "max" when the cgroup has no limit
    return int(value) if value.isdigit() else None


def _log_in_flight(memory_current: int, memory_max: int) -> None:
    now = time.monotonic()
    with _in_flight_lock:
        requests = sorted(_in_flight.values(), key=lambda request: request.arrow_bytes or 0, reverse=True)
    lines = [
        f"{request.description} for {now - request.started_at:.1f}s, arrow table of "
        f"{'unknown' if request.arrow_bytes is None else request.arrow_bytes} bytes"
        for request in requests
    ]
    logging.warning(
        f"rows memory at {memory_current / memory_max:.0%} of the pod limit ({memory_current} of {memory_max} bytes),"
        f" worker {os.getpid()} serves {len(requests)} requests: " + "; ".join(lines)
    )


def _watch(threshold: float, interval_seconds: float) -> None:
    last_log = float("-inf")
    while True:
        time.sleep(interval_seconds)
        if _release_requested.is_set():
            _release_requested.clear()
            _release_freed_memory()
        memory_current = _read_int(CGROUP_MEMORY_CURRENT)
        memory_max = _read_int(CGROUP_MEMORY_MAX)
        if memory_current is None or memory_max is None:
            continue
        now = time.monotonic()
        if memory_current >= threshold * memory_max and now - last_log >= WATCHDOG_LOG_INTERVAL_SECONDS:
            _log_in_flight(memory_current, memory_max)
            last_log = now


def start_memory_watchdog(
    threshold: float = WATCHDOG_THRESHOLD, interval_seconds: float = WATCHDOG_INTERVAL_SECONDS
) -> None:
    """Log the in-flight requests of this worker when the pod memory gets close to its limit.

    It also gives the memory freed by a request that grew the worker back to the OS.
    """
    if _read_int(CGROUP_MEMORY_MAX) is None:
        logging.info("no cgroup memory limit to watch, the rows memory watchdog is not started")
        return
    threading.Thread(
        target=_watch, args=(threshold, interval_seconds), name="rows-memory-watchdog", daemon=True
    ).start()
