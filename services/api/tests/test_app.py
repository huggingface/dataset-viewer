# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
from http import HTTPStatus
from typing import Dict, Optional

import pytest
from libcache.simple_cache import _clean_database as _clean_cache_database
from libcache.simple_cache import upsert_first_rows_response, upsert_splits_response
from libqueue.queue import Queue, _clean_queue_database
from pytest_httpserver import HTTPServer
from starlette.testclient import TestClient

from api.app import create_app
from api.utils import JobType

from .utils import auth_callback


@pytest.fixture(scope="module")
def client(monkeypatch_session: pytest.MonkeyPatch) -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_databases() -> None:
    _clean_cache_database()
    _clean_queue_database()


splits_queue = Queue(type=JobType.SPLITS.value)


def test_cors(client: TestClient) -> None:
    origin = "http://localhost:3000"
    method = "GET"
    header = "X-Requested-With"
    response = client.options(
        "/splits?dataset=dataset1",
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
    json = response.json()
    assert "valid" in json


# caveat: the returned status codes don't simulate the reality
# they're just used to check every case
@pytest.mark.parametrize(
    "headers,status_code,error_code",
    [
        ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
        ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
        ({}, 200, None),
    ],
)
def test_is_valid_auth(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    headers: Dict[str, str],
    status_code: int,
    error_code: Optional[str],
) -> None:
    dataset = "dataset-which-does-not-exist"
    httpserver.expect_request(hf_auth_path % dataset, headers=headers).respond_with_handler(auth_callback)
    response = client.get(f"/is-valid?dataset={dataset}", headers=headers)
    assert response.status_code == status_code
    assert response.headers.get("X-Error-Code") == error_code


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_get_splits(client: TestClient) -> None:
    # missing parameter
    response = client.get("/splits")
    assert response.status_code == 422
    # empty parameter
    response = client.get("/splits?dataset=")
    assert response.status_code == 422


# caveat: the returned status codes don't simulate the reality
# they're just used to check every case
@pytest.mark.parametrize(
    "headers,status_code,error_code",
    [
        ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
        ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
        ({}, 404, "SplitsResponseNotFound"),
    ],
)
def test_splits_auth(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    headers: Dict[str, str],
    status_code: int,
    error_code: str,
) -> None:
    dataset = "dataset-which-does-not-exist"
    httpserver.expect_request(hf_auth_path % dataset, headers=headers).respond_with_handler(auth_callback)
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({}), headers={"X-Error-Code": "RepoNotFound"}
    )
    response = client.get(f"/splits?dataset={dataset}", headers=headers)
    assert response.status_code == status_code, f"{response.headers}, {response.json()}"
    assert response.headers.get("X-Error-Code") == error_code


@pytest.mark.parametrize(
    "dataset,config,split",
    [
        (None, None, None),
        ("a", None, None),
        ("a", "b", None),
        ("a", "b", ""),
    ],
)
def test_get_first_rows_missing_parameter(
    client: TestClient, dataset: Optional[str], config: Optional[str], split: Optional[str]
) -> None:
    response = client.get("/first-rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "exists,is_private,expected_error_code",
    [
        (False, None, "ExternalAuthenticatedError"),
        (True, True, "SplitsResponseNotFound"),
        (True, False, "SplitsResponseNotReady"),
    ],
)
def test_splits_cache_refreshing(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    exists: bool,
    is_private: Optional[bool],
    expected_error_code: str,
) -> None:
    dataset = "dataset-to-be-processed"
    httpserver.expect_request(hf_auth_path % dataset).respond_with_data(status=200 if exists else 404)
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({"private": is_private}), headers={} if exists else {"X-Error-Code": "RepoNotFound"}
    )

    response = client.get("/splits", params={"dataset": dataset})
    assert response.headers["X-Error-Code"] == expected_error_code

    if expected_error_code == "SplitsResponseNotReady":
        # a subsequent request should return the same error code
        response = client.get("/splits", params={"dataset": dataset})
        assert response.headers["X-Error-Code"] == expected_error_code

        # simulate the worker
        upsert_splits_response(dataset, {"key": "value"}, HTTPStatus.OK)
        response = client.get("/splits", params={"dataset": dataset})
        assert response.json()["key"] == "value"
        assert response.status_code == 200


@pytest.mark.parametrize(
    "exists,is_private,expected_error_code",
    [
        (False, None, "ExternalAuthenticatedError"),
        (True, True, "FirstRowsResponseNotFound"),
        (True, False, "FirstRowsResponseNotReady"),
    ],
)
def test_first_rows_cache_refreshing(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    exists: bool,
    is_private: Optional[bool],
    expected_error_code: str,
) -> None:
    dataset = "dataset-to-be-processed"
    config = "default"
    split = "train"
    httpserver.expect_request(hf_auth_path % dataset).respond_with_data(status=200 if exists else 404)
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({"private": is_private}), headers={} if exists else {"X-Error-Code": "RepoNotFound"}
    )

    response = client.get("/first-rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.headers["X-Error-Code"] == expected_error_code

    if expected_error_code == "FirstRowsResponseNotReady":
        # a subsequent request should return the same error code
        response = client.get("/first-rows", params={"dataset": dataset, "config": config, "split": split})
        assert response.headers["X-Error-Code"] == expected_error_code

        # simulate the worker
        upsert_first_rows_response(dataset, config, split, {"key": "value"}, HTTPStatus.OK)
        response = client.get("/first-rows", params={"dataset": dataset, "config": config, "split": split})
        assert response.json()["key"] == "value"
        assert response.status_code == 200


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


@pytest.mark.parametrize(
    "payload,exists_on_the_hub,expected_status,expected_is_updated",
    [
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, True, 200, True),
        (
            {
                "event": "move",
                "movedTo": "webhook-test",
                "repo": {"type": "dataset", "name": "previous-name", "gitalyUid": "123"},
            },
            True,
            200,
            True,
        ),
        (
            {"event": "doesnotexist", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}},
            True,
            400,
            False,
        ),
        (
            {"event": "add", "repo": {"type": "dataset", "name": "webhook-test"}},
            True,
            200,
            True,
        ),
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, False, 200, False),
    ],
)
def test_webhook(
    client: TestClient,
    httpserver: HTTPServer,
    payload: Dict,
    exists_on_the_hub: bool,
    expected_status: int,
    expected_is_updated: bool,
) -> None:
    dataset = "webhook-test"
    headers = None if exists_on_the_hub else {"X-Error-Code": "RepoNotFound"}
    status = 200 if exists_on_the_hub else 404
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({"private": False}), headers=headers, status=status
    )
    response = client.post("/webhook", json=payload)
    assert response.status_code == expected_status, response.text
    assert splits_queue.is_job_in_process(dataset=dataset) is expected_is_updated
