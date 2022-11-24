# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest
from libcache.simple_cache import _clean_cache_database
from libcommon.processing_steps import PROCESSING_STEPS
from libqueue.queue import _clean_queue_database
from starlette.testclient import TestClient

from admin.app import create_app
from admin.config import AppConfig


@pytest.fixture(scope="module")
def client(monkeypatch_session: pytest.MonkeyPatch) -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()
    _clean_queue_database()


def test_cors(client: TestClient) -> None:
    origin = "http://localhost:3000"
    method = "GET"
    header = "X-Requested-With"
    response = client.options(
        "/pending-jobs",
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
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    lines = text.split("\n")
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}

    # the middleware should have recorded the request
    name = 'starlette_requests_total{method="GET",path_template="/metrics"}'
    assert name in metrics, metrics
    assert metrics[name] > 0, metrics


def test_pending_jobs(client: TestClient) -> None:
    response = client.get("/pending-jobs")
    assert response.status_code == 200
    json = response.json()
    for processing_step in PROCESSING_STEPS:
        assert json[processing_step.job_type] == {"waiting": [], "started": []}


@pytest.mark.parametrize(
    "cursor,http_status,error_code",
    [
        (None, 200, None),
        ("", 200, None),
        ("invalid cursor", 422, "InvalidParameter"),
    ],
)
def test_cache_reports(client: TestClient, cursor: Optional[str], http_status: int, error_code: Optional[str]) -> None:
    path = PROCESSING_STEPS[0].endpoint
    cursor_str = f"?cursor={cursor}" if cursor else ""
    response = client.get(f"/cache-reports{path}{cursor_str}")
    assert response.status_code == http_status
    if error_code:
        assert isinstance(response.json()["error"], str)
        assert response.headers["X-Error-Code"] == error_code
    else:
        assert response.json() == {"cache_reports": [], "next_cursor": ""}
        assert "X-Error-Code" not in response.headers
