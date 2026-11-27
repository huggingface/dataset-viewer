# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import anyio
import pyarrow.parquet as pq
from libapi.exceptions import ResponseNotReadyError
from libcommon.parquet_utils import ParquetFileMetadataItem


class ArrowBudget:
    """The bytes of Arrow data that the /rows requests of a worker hold at once.

    A response takes 5 to 7 times the size of its Arrow table while it is built, and a worker serves several requests
    at once: without a budget, a burst of big tables outgrows the pod memory. The requests get their bytes in the
    order they ask for them, and a request that asks for more than the whole budget gets them when it is alone.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.held = 0
        self._waiters: deque[tuple[int, anyio.Event]] = deque()

    def _fits(self, num_bytes: int) -> bool:
        return self.held == 0 or self.held + num_bytes <= self.max_bytes

    def _wake_up_waiters(self) -> None:
        while self._waiters and self._fits(self._waiters[0][0]):
            num_bytes, event = self._waiters.popleft()
            self.held += num_bytes
            event.set()

    @asynccontextmanager
    async def hold(self, num_bytes: int, wait_seconds: float) -> AsyncIterator[Callable[[int], None]]:
        """Hold num_bytes of the budget, waiting for them up to wait_seconds.

        It yields a function that raises the bytes held to the real size of the table, once it is known: the estimate
        can be too low, for example for the columns stored with a dictionary.
        """
        if not self._waiters and self._fits(num_bytes):
            self.held += num_bytes
        else:
            waiter = (num_bytes, anyio.Event())
            self._waiters.append(waiter)
            try:
                with anyio.fail_after(wait_seconds):
                    await waiter[1].wait()
            except BaseException as err:
                if waiter[1].is_set():
                    self.held -= num_bytes
                else:
                    self._waiters.remove(waiter)
                self._wake_up_waiters()
                if isinstance(err, TimeoutError):
                    raise ResponseNotReadyError(
                        "The server is busier than usual and the response is not ready yet. Please retry later."
                    ) from None
                raise
        held = num_bytes

        def hold_at_least(real_num_bytes: int) -> None:
            nonlocal held
            if real_num_bytes > held:
                self.held += real_num_bytes - held
                held = real_num_bytes

        try:
            yield hold_at_least
        finally:
            self.held -= held
            self._wake_up_waiters()


@lru_cache(maxsize=16)
def _row_groups(parquet_metadata_directory: Path, subpaths: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """The number of rows and the uncompressed size of each row group of the split, in order."""
    row_groups = []
    for subpath in subpaths:
        metadata = pq.read_metadata(parquet_metadata_directory / subpath)
        for index in range(metadata.num_row_groups):
            row_group = metadata.row_group(index)
            row_groups.append((row_group.num_rows, row_group.total_byte_size))
    return tuple(row_groups)


def estimate_arrow_bytes(
    parquet_files: list[ParquetFileMetadataItem], parquet_metadata_directory: Path, offset: int, length: int
) -> int:
    """Estimate the size of the Arrow table of the rows [offset, offset + length) from the parquet metadata."""
    row_groups = _row_groups(parquet_metadata_directory, tuple(f["parquet_metadata_subpath"] for f in parquet_files))
    estimate = 0
    start = 0
    for num_rows, num_bytes in row_groups:
        overlap = min(offset + length, start + num_rows) - max(offset, start)
        if overlap > 0:
            estimate += num_bytes * overlap // num_rows
        start += num_rows
    return estimate
