# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import asyncio
from http import HTTPStatus
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pytest
from datasets import Features, Value
from libcommon.parquet_utils import EmptyParquetMetadataError, TooBigRows
from libcommon.storage_client import StorageClient
from starlette.requests import Request

from rows.routes import rows
from rows.routes.rows import create_rows_endpoint


class FakeRowsIndex:
    revision = "revision"
    features = Features({"a": Value("int64")})
    partial = False
    num_rows_total = 1
    running = 0
    max_running = 0

    async def scan_size(self, offset: int, length: int) -> int:
        return 1

    async def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        FakeRowsIndex.running += 1
        FakeRowsIndex.max_running = max(FakeRowsIndex.max_running, FakeRowsIndex.running)
        await asyncio.sleep(0.2)
        FakeRowsIndex.running -= 1
        return pa.table({"a": [offset]}).slice(0, length), []


def request(split: str = "train") -> Request:
    query = f"dataset=ds&config=default&split={split}&offset=0&length=1".encode()
    return Request({"type": "http", "method": "GET", "path": "/rows", "query_string": query, "headers": []})


def endpoint(tmp_path: Any, max_concurrent_queries: int = 8, **kwargs: Any) -> Any:
    return create_rows_endpoint(
        cached_assets_storage_client=StorageClient(
            protocol="file", storage_root=str(tmp_path / "assets"), base_url="http://localhost/assets"
        ),
        parquet_metadata_directory=tmp_path,
        max_arrow_data_in_memory=10**9,
        max_concurrent_queries=max_concurrent_queries,
        max_arrow_data_per_worker=10**9,
        hf_endpoint="http://localhost",
        blocked_datasets=[],
        **kwargs,
    )


async def no_auth_check(**_: Any) -> None:
    return None


@pytest.mark.anyio
async def test_queries_wait_for_a_slot(tmp_path: Any) -> None:
    with (
        patch.object(rows, "RowsIndex", lambda **_: FakeRowsIndex()),
        patch.object(rows, "auth_check", no_auth_check),
    ):
        rows_endpoint = endpoint(tmp_path, max_concurrent_queries=3)
        responses = await asyncio.gather(*(rows_endpoint(request()) for _ in range(10)))

    assert [response.status_code for response in responses] == [HTTPStatus.OK] * 10
    assert FakeRowsIndex.max_running == 3


@pytest.mark.anyio
async def test_a_request_that_waits_too_long_for_a_slot_is_not_ready(tmp_path: Any) -> None:
    with (
        patch.object(rows, "RowsIndex", lambda **_: FakeRowsIndex()),
        patch.object(rows, "auth_check", no_auth_check),
        patch.object(rows, "QUERY_SLOT_WAIT_SECONDS", 0.05),
    ):
        rows_endpoint = endpoint(tmp_path, max_concurrent_queries=1)
        responses = await asyncio.gather(*(rows_endpoint(request()) for _ in range(2)))

    assert sorted(response.headers.get("X-Error-Code", "") for response in responses) == ["", "ResponseNotReady"]


@pytest.mark.anyio
async def test_a_split_without_parquet_files_is_not_found(tmp_path: Any) -> None:
    def missing_split(**_: Any) -> FakeRowsIndex:
        raise EmptyParquetMetadataError("No parquet files found.")

    with patch.object(rows, "RowsIndex", missing_split), patch.object(rows, "auth_check", no_auth_check):
        response = await endpoint(tmp_path)(request(split="does-not-exist"))

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["X-Error-Code"] == "ResponseNotFound"


class TooBigRowsIndex(FakeRowsIndex):
    async def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        raise TooBigRows("Scan size limit exceeded: attempted to read 410817512 bytes, limit is 300000000 bytes")


@pytest.mark.anyio
async def test_a_scan_over_the_limit_is_a_client_error_cached_long(tmp_path: Any) -> None:
    with (
        patch.object(rows, "RowsIndex", lambda **_: TooBigRowsIndex()),
        patch.object(rows, "auth_check", no_auth_check),
    ):
        response = await endpoint(tmp_path, max_age_long=120, max_age_short=10)(request())

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.headers["X-Error-Code"] == "TooBigContentError"
    assert response.headers["Cache-Control"] == "max-age=120"
