# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import io
import os
import signal
from http import HTTPStatus
from typing import Any, Optional
from unittest.mock import patch

import pyarrow as pa
import pytest
from datasets import Features, Image
from libapi.exceptions import TooBigContentError, TransformRowsProcessingError
from libcommon.dtos import PaginatedResponse
from PIL import Image as PILImage
from prometheus_client import REGISTRY

from rows.config import AppConfig
from rows.media_child import MediaChildProcess, create_cached_assets_storage_client
from rows.routes import rows
from rows.routes.rows import create_rows_endpoint

from .test_rows_limits import FakeRowsIndex, no_auth_check, request

IMAGE_FEATURES = Features({"image": Image()})


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def image_table(image: bytes) -> pa.Table:
    return pa.Table.from_pydict({"image": [{"bytes": image, "path": None}]}, schema=IMAGE_FEATURES.arrow_schema)


async def build(child: MediaChildProcess, image: bytes) -> PaginatedResponse:
    return await child.create_response(
        description="test",
        dataset="ds",
        revision="revision",
        config="default",
        split="train",
        pa_table=image_table(image),
        offset=0,
        features=IMAGE_FEATURES,
        num_rows_total=1,
        partial=False,
    )


def stops(reason: str) -> float:
    return REGISTRY.get_sample_value("rows_media_child_stops_total", {"reason": reason}) or 0


def child_pid(child: MediaChildProcess) -> Optional[int]:
    return None if child._child is None else child._child.pid


@pytest.mark.anyio
async def test_the_child_builds_the_media_and_serves_the_next_requests(app_config: AppConfig) -> None:
    child = MediaChildProcess(app_config=app_config)
    try:
        response = await build(child, png_bytes())
        pid = child_pid(child)
        await build(child, png_bytes())
    finally:
        child.stop()

    image = response["rows"][0]["row"]["image"]
    assert image["src"].startswith(f"{app_config.cached_assets.base_url}/ds/--/revision/--/default/train/0/image/")
    assert (image["height"], image["width"]) == (4, 4)
    assert pid is not None and pid != os.getpid()
    assert child_pid(child) is None


@pytest.mark.anyio
async def test_the_error_of_a_response_is_raised_and_the_child_kept(app_config: AppConfig) -> None:
    child = MediaChildProcess(app_config=app_config)
    try:
        await build(child, png_bytes())
        pid = child_pid(child)
        with pytest.raises(TransformRowsProcessingError):
            await build(child, b"not an image")
        assert child_pid(child) == pid
    finally:
        child.stop()


@pytest.mark.anyio
async def test_the_child_is_killed_above_the_memory_threshold(app_config: AppConfig) -> None:
    killed = stops("killed")
    # a fresh interpreter holds more than 1 MiB before it answers its first request
    child = MediaChildProcess(app_config=app_config, kill_memory_bytes=2**20)
    try:
        with pytest.raises(TooBigContentError):
            await build(child, png_bytes())
        assert stops("killed") == killed + 1
        child.kill_memory_bytes = None
        response = await build(child, png_bytes())
    finally:
        child.stop()

    assert response["rows"][0]["row"]["image"]["src"]


@pytest.mark.anyio
async def test_the_child_is_replaced_above_the_idle_memory_threshold(app_config: AppConfig) -> None:
    replaced = stops("replaced")
    child = MediaChildProcess(app_config=app_config, max_idle_memory_bytes=2**20)
    try:
        await build(child, png_bytes())
        first_pid = child_pid(child)
        await build(child, png_bytes())
        second_pid = child_pid(child)
    finally:
        child.stop()

    assert stops("replaced") == replaced + 2
    assert len({first_pid, second_pid}) == 2


@pytest.mark.anyio
async def test_a_child_that_died_while_idle_is_replaced(app_config: AppConfig) -> None:
    crashed = stops("crashed")
    child = MediaChildProcess(app_config=app_config)
    try:
        await build(child, png_bytes())
        pid = child_pid(child)
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        if child._child is not None:
            child._child.process.join(10)
        response = await build(child, png_bytes())
        assert child_pid(child) != pid
    finally:
        child.stop()

    assert response["rows"][0]["row"]["image"]["src"]
    assert stops("crashed") == crashed + 1


class RecordingChild(MediaChildProcess):
    def __init__(self, app_config: AppConfig) -> None:
        super().__init__(app_config=app_config)
        self.calls = 0

    async def create_response(self, **kwargs: Any) -> PaginatedResponse:  # type: ignore[override]
        self.calls += 1
        return {"features": [], "rows": [], "num_rows_total": 1, "num_rows_per_page": 100, "partial": False}


class ImageRowsIndex(FakeRowsIndex):
    features = IMAGE_FEATURES

    async def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        return image_table(png_bytes()), []


@pytest.mark.anyio
@pytest.mark.parametrize("rows_index,calls", [(FakeRowsIndex, 0), (ImageRowsIndex, 1)])
async def test_only_the_requests_with_media_go_to_the_child(
    app_config: AppConfig, tmp_path: Any, rows_index: type[FakeRowsIndex], calls: int
) -> None:
    child = RecordingChild(app_config=app_config)
    with patch.object(rows, "RowsIndex", lambda **_: rows_index()), patch.object(rows, "auth_check", no_auth_check):
        rows_endpoint = create_rows_endpoint(
            cached_assets_storage_client=create_cached_assets_storage_client(app_config),
            parquet_metadata_directory=tmp_path,
            max_arrow_data_in_memory=10**9,
            max_concurrent_queries=8,
            hf_endpoint="http://localhost",
            blocked_datasets=[],
            media_child=child,
        )
        response = await rows_endpoint(request())

    assert response.status_code == HTTPStatus.OK
    assert child.calls == calls
