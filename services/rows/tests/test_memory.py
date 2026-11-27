# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import asyncio
import logging
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pytest
from datasets import Features, Value

from rows import memory
from rows.routes import rows

from .test_rows_limits import endpoint, no_auth_check, request


def write_cgroup(tmp_path: Path, current: int, limit: str) -> None:
    (tmp_path / "memory.current").write_text(f"{current}\n")
    (tmp_path / "memory.max").write_text(f"{limit}\n")


@pytest.fixture
def cgroup(tmp_path: Path) -> Any:
    with (
        patch.object(memory, "CGROUP_MEMORY_CURRENT", tmp_path / "memory.current"),
        patch.object(memory, "CGROUP_MEMORY_MAX", tmp_path / "memory.max"),
    ):
        yield tmp_path


class BigRowsIndex:
    """A query that keeps a table of 8 MB alive while the watchdog looks at the in-flight requests."""

    revision = "revision"
    parquet_files: list[Any] = []
    features = Features({"a": Value("int64")})
    partial = False
    num_rows_total = 1_000_000

    async def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        return pa.table({"a": list(range(1_000_000))}), []


@pytest.mark.anyio
async def test_the_watchdog_logs_the_in_flight_requests_close_to_the_limit(
    cgroup: Path, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    write_cgroup(cgroup, current=95, limit="100")
    release = asyncio.Event()

    async def slow_response(**_: Any) -> dict[str, Any]:
        await release.wait()
        return {"rows": []}

    with (
        patch.object(rows, "RowsIndex", lambda **_: BigRowsIndex()),
        patch.object(rows, "auth_check", no_auth_check),
        patch.object(rows, "create_response", slow_response),
        caplog.at_level(logging.INFO),
    ):
        pending = asyncio.ensure_future(endpoint(tmp_path)(request()))
        while not memory._in_flight:
            await asyncio.sleep(0.01)
        memory._log_in_flight(memory_current=95, memory_max=100)
        release.set()
        response = await pending

    assert response.status_code == HTTPStatus.OK
    assert "/rows arrow table of 8000000 bytes" in caplog.text
    warning = next(record for record in caplog.records if record.levelno == logging.WARNING)
    assert "95% of the pod limit" in warning.message
    assert "dataset='ds'" in warning.message and "arrow table of 8000000 bytes" in warning.message
    assert not memory._in_flight, "a finished request is no longer in flight"


def test_the_watchdog_does_not_start_without_a_memory_limit(cgroup: Path) -> None:
    write_cgroup(cgroup, current=10, limit="max")
    with patch("rows.memory.threading.Thread") as thread:
        memory.start_memory_watchdog()
    thread.assert_not_called()


def test_a_request_that_grows_the_worker_memory_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    readings = iter([100 * 2**20, 100 * 2**20 + memory.MEMORY_GROWTH_LOG_BYTES, 0, 1])
    with (
        patch.object(memory, "_read_anonymous_memory", lambda: next(readings)),
        caplog.at_level(logging.WARNING),
    ):
        with memory.track_request("dataset='big'") as request:
            memory.record_arrow_bytes(request, 1000)
        with memory.track_request("dataset='small'"):
            pass

    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "256 MiB more anonymous memory (356 MiB) after dataset='big'" in warnings[0]
    assert "arrow table of 1000 bytes, with 0 other requests in flight" in warnings[0]
    assert memory._release_requested.is_set(), "the watchdog gives the freed memory back to the OS"
    memory._release_requested.clear()


def test_the_watchdog_gives_the_freed_memory_back_to_the_os(caplog: pytest.LogCaptureFixture) -> None:
    calls = []
    readings = iter([600 * 2**20, 200 * 2**20])

    def malloc_trim(pad: int) -> int:
        calls.append("malloc_trim")
        return 1

    with (
        patch.object(memory, "_read_anonymous_memory", lambda: next(readings)),
        patch.object(memory, "_MALLOC_TRIM", malloc_trim),
        patch("rows.memory.gc.collect", lambda: calls.append("gc.collect")),
        patch.object(memory, "_read_int", lambda path: None),
        patch("rows.memory.time.sleep", side_effect=[None, StopIteration]),
        caplog.at_level(logging.INFO),
    ):
        memory._release_requested.set()
        with pytest.raises(StopIteration):
            memory._watch(threshold=0.85, interval_seconds=1.0)

    assert calls == ["gc.collect", "malloc_trim"]
    assert not memory._release_requested.is_set()
    assert "gave 400 MiB back to the OS, it holds 200 MiB of anonymous memory" in caplog.text
