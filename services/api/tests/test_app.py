# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest
from starlette.testclient import TestClient

from api.app import create_app_with_config
from api.config import AppConfig, EndpointConfig


@pytest.fixture(scope="module")
def client(
    monkeypatch_session: pytest.MonkeyPatch, app_config: AppConfig, endpoint_config: EndpointConfig
) -> TestClient:
    return TestClient(create_app_with_config(app_config=app_config, endpoint_config=endpoint_config))


def test_cors(client: TestClient, first_dataset_endpoint: str) -> None:
    origin = "http://localhost:3000"
    method = "GET"
    header = "X-Requested-With"
    response = client.options(
        f"{first_dataset_endpoint}?dataset=dataset1",
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


def test_get_valid_datasets(client: TestClient) -> None:
    response = client.get("/valid")
    assert response.status_code == 200
    assert "valid" in response.json()


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_get_endpoint(client: TestClient, first_dataset_endpoint: str) -> None:
    # missing parameter
    response = client.get(first_dataset_endpoint)
    assert response.status_code == 422
    # empty parameter
    response = client.get(f"{first_dataset_endpoint}?dataset=")
    assert response.status_code == 422


@pytest.mark.parametrize(
    "dataset,config",
    [
        (None, None),
        ("a", None),
        ("a", ""),
    ],
)
def test_get_config_missing_parameter(
    client: TestClient,
    dataset: Optional[str],
    config: Optional[str],
    first_config_endoint: str,
) -> None:
    response = client.get(first_config_endoint, params={"dataset": dataset, "config": config, "split": None})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "dataset,config,split",
    [
        (None, None, None),
        ("a", None, None),
        ("a", "b", None),
        ("a", "b", ""),
    ],
)
def test_get_split_missing_parameter(
    client: TestClient,
    dataset: Optional[str],
    config: Optional[str],
    split: Optional[str],
    first_split_endpoint: str,
) -> None:
    response = client.get(first_split_endpoint, params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 422


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
