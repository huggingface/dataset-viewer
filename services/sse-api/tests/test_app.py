# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import httpx
import pytest
from httpx_sse import aconnect_sse
from starlette.applications import Starlette
from starlette.testclient import TestClient

from sse_api.app import create_app_with_config
from sse_api.config import AppConfig


@pytest.fixture(scope="module")
def app(monkeypatch_session: pytest.MonkeyPatch, app_config: AppConfig) -> Starlette:
    return create_app_with_config(app_config=app_config)


@pytest.fixture(scope="module")
def client(app: Starlette) -> TestClient:
    return TestClient(app)


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_metrics(client: TestClient) -> None:
    response = client.get("/healthcheck")
    response = client.get("/metrics")
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


WEIRD_MANDATORY_PREFIX = "http://localhost:8000"


@pytest.mark.asyncio
async def test_numbers(app: Starlette) -> None:
    async with httpx.AsyncClient(app=app) as client:
        async with aconnect_sse(client, f"{WEIRD_MANDATORY_PREFIX}/numbers") as event_source:
            event_iter = event_source.aiter_sse()
            for i in range(1, 5):
                event = await event_iter.__anext__()
                # event = await anext(event_iter)
                # ^ only available in 3.10
                print(event.data)
                assert event.event == "message", event.data
                assert event.data == str(i), event.data
