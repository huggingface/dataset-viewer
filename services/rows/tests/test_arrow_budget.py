# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from pathlib import Path
from typing import Any

import anyio
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from libapi.exceptions import ResponseNotReadyError

from rows.arrow_budget import ArrowBudget, estimate_arrow_bytes


@pytest.mark.anyio
async def test_the_requests_over_the_budget_wait_in_their_order() -> None:
    budget = ArrowBudget(max_bytes=100)
    order: list[str] = []

    async def request(name: str, num_bytes: int, delay: float) -> None:
        await anyio.sleep(delay)
        async with budget.hold(num_bytes, wait_seconds=5):
            order.append(f"start {name}")
            await anyio.sleep(0.1)
            order.append(f"end {name}")

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(request, "a", 60, 0)
        # waits for a, and the smaller c, which comes after it, waits for it
        tasks.start_soon(request, "b", 60, 0.01)
        tasks.start_soon(request, "c", 10, 0.02)
        # bigger than the whole budget: it waits until it is alone
        tasks.start_soon(request, "d", 500, 0.03)

    assert order == ["start a", "end a", "start b", "start c", "end b", "end c", "start d", "end d"]
    assert budget.held == 0


@pytest.mark.anyio
async def test_a_request_that_waits_too_long_for_the_budget_is_not_ready() -> None:
    budget = ArrowBudget(max_bytes=100)
    async with budget.hold(80, wait_seconds=1):
        with pytest.raises(ResponseNotReadyError):
            async with budget.hold(80, wait_seconds=0.05):
                pass
        async with budget.hold(20, wait_seconds=0.05):
            assert budget.held == 100
    assert budget.held == 0


@pytest.mark.anyio
async def test_the_budget_follows_the_real_size_of_the_table() -> None:
    budget = ArrowBudget(max_bytes=100)
    async with budget.hold(10, wait_seconds=1) as hold_at_least:
        hold_at_least(70)
        hold_at_least(30)
        assert budget.held == 70
    assert budget.held == 0


def test_the_size_of_the_rows_is_estimated_from_the_parquet_metadata(tmp_path: Path) -> None:
    files: list[Any] = []
    for name in ["0000.parquet", "0001.parquet"]:
        pq.write_table(pa.table({"text": ["x" * 1000] * 100}), tmp_path / name, row_group_size=10)
        files.append({"parquet_metadata_subpath": name})
    one_row = pq.read_metadata(tmp_path / "0000.parquet").row_group(0).total_byte_size // 10

    assert estimate_arrow_bytes(files, tmp_path, offset=0, length=1) == pytest.approx(one_row, abs=1)
    # across the two files
    assert estimate_arrow_bytes(files, tmp_path, offset=95, length=10) == pytest.approx(10 * one_row, abs=10)
    assert estimate_arrow_bytes(files, tmp_path, offset=195, length=100) == pytest.approx(5 * one_row, abs=5)
    assert estimate_arrow_bytes(files, tmp_path, offset=500, length=100) == 0
