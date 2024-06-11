# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import pytest
from starlette.testclient import TestClient

from webhook.app import create_app_with_config
from webhook.config import AppConfig


@pytest.fixture(scope="module")
def client(monkeypatch_session: pytest.MonkeyPatch, app_config: AppConfig) -> TestClient:
    return TestClient(create_app_with_config(app_config=app_config))


def test_cors(client: TestClient) -> None:
    origin = "http://localhost:3000"
    method = "GET"
    header = "X-Requested-With"
    response = client.options(
        "/webhook",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": header,
        },
    )
    assert response.status_code == 200
    assert (
        origin in [o.strip() for o in response.headers["Access-Control-Allow-Origin"].split(",")]
        or response.headers["Access-Control-Allow-Origin"] == "*"
    )
    assert (
        header in [o.strip() for o in response.headers["Access-Control-Allow-Headers"].split(",")]
        or response.headers["Access-Control-Expose-Headers"] == "*"
    )
    assert (
        method in [o.strip() for o in response.headers["Access-Control-Allow-Methods"].split(",")]
        or response.headers["Access-Control-Expose-Headers"] == "*"
    )
    assert response.headers["Access-Control-Allow-Credentials"] == "true"


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
    starlette_requests_metric = 'starlette_requests_total{method="GET",path_template="/metrics"}'
    steps_processing_time_metric = 'method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"}'
    for name in [starlette_requests_metric, steps_processing_time_metric]:
        assert name in metrics, metrics
        assert metrics[name] > 0, metrics
