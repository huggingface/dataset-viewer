# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
import uvicorn
from httpx_sse import aconnect_sse
from starlette.applications import Starlette

from sse_api.app import create_app_with_config
from sse_api.config import AppConfig

from .uvicorn_server import UvicornServer


@pytest_asyncio.fixture(scope="function")
async def app_test(
    app_config: AppConfig, event_loop: asyncio.events.AbstractEventLoop
) -> AsyncGenerator[Starlette, None]:
    app = create_app_with_config(app_config)
    config = uvicorn.Config(app=app, port=5555, log_level="warning", loop="asyncio")  # event_loop)

    server = UvicornServer(config)
    await server.start()
    try:
        yield app
    finally:
        await server.shutdown()


@pytest_asyncio.fixture(scope="function")
async def client(app_test: Starlette) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(base_url=APP_HOST) as client:
        yield client


APP_HOST = "http://localhost:5555"


@pytest.mark.asyncio
async def test_provided_loop_is_running_loop(event_loop: asyncio.events.AbstractEventLoop) -> None:
    assert event_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_get_healthcheck(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


@pytest.mark.asyncio
async def test_metrics(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    lines = text.split("\n")
    # examples:
    # starlette_requests_total{method="GET",path_template="/metrics"} 1.0
    # method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"} 1.6772013623267412e-05
    metrics = {
        parts[0]: float(parts[1]) for line in lines if line and line[0] != "#" and (parts := line.rsplit(" ", 1))
    }

    # the metrics should contain at least the following
    for name in [
        'starlette_requests_total{method="GET",path_template="/metrics"}',
        'method_steps_processing_time_seconds_sum{context="None",method="healthcheck_endpoint",step="all"}',
    ]:
        assert name in metrics, metrics
        assert metrics[name] > 0, metrics


@pytest.mark.asyncio
async def test_numbers(client: httpx.AsyncClient) -> None:
    async with aconnect_sse(client, f"{APP_HOST}/numbers") as event_source:
        event_iter = event_source.aiter_sse()
        for _ in range(1, 5):
            event = await event_iter.__anext__()
            # event = await anext(event_iter)
            # ^ only available in 3.10
            assert event.event == "message", event.data
            assert int(event.data) >= 0, event.data
            assert int(event.data) <= 100, event.data
    await client.aclose()
