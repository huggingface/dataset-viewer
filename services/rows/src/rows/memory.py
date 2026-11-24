# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

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

from prometheus_client import Histogram

# the kernel kills the pod without any warning when its cgroup reaches memory.max: the watchdog logs what the workers
# are doing when the memory gets close to it, which is the only trace left of the request that caused the OOM
CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
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


@contextmanager
def track_request(description: str) -> Iterator[InFlightRequest]:
    request = InFlightRequest(description=description)
    request_id = next(_ids)
    with _in_flight_lock:
        _in_flight[request_id] = request
    try:
        yield request
    finally:
        with _in_flight_lock:
            del _in_flight[request_id]


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
    """Log the in-flight requests of this worker when the pod memory gets close to its limit."""
    if _read_int(CGROUP_MEMORY_MAX) is None:
        logging.info("no cgroup memory limit to watch, the rows memory watchdog is not started")
        return
    threading.Thread(
        target=_watch, args=(threshold, interval_seconds), name="rows-memory-watchdog", daemon=True
    ).start()
