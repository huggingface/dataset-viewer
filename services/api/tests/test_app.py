from http import HTTPStatus
from typing import Dict, Optional

import pytest
import responses

# from libcache.cache import clean_database as clean_cache_database
from libcache.cache import clean_database as clean_cache_database
from libcache.cache import (
    create_or_mark_dataset_as_stale,
    create_or_mark_split_as_stale,
)
from libcache.simple_cache import (
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
    upsert_first_rows_response,
    upsert_splits_response,
)
from libqueue.queue import (
    add_dataset_job,
    add_first_rows_job,
    add_split_job,
    add_splits_job,
)
from libqueue.queue import clean_database as clean_queue_database
from starlette.testclient import TestClient

from api.app import create_app
from api.config import EXTERNAL_AUTH_URL, MONGO_QUEUE_DATABASE

from .utils import request_callback

external_auth_url = EXTERNAL_AUTH_URL or "%s"  # for mypy


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    # if "test" not in MONGO_CACHE_DATABASE:
    #     raise ValueError("Tests on cache must be launched on a test mongo database")
    if "test" not in MONGO_QUEUE_DATABASE:
        raise ValueError("Tests on queue must be launched on a test mongo database")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_databases() -> None:
    clean_cache_database()
    clean_queue_database()


# TODO: move to e2e tests
# def test_get_cache_reports(client: TestClient) -> None:
#     refresh_dataset_split_full_names("acronym_identification")
#     response = client.get("/cache-reports")
#     assert response.status_code == 200
#     json = response.json()
#     assert "datasets" in json
#     assert "splits" in json
#     datasets = json["datasets"]
#     assert "empty" in datasets
#     assert "error" in datasets
#     assert "stale" in datasets
#     assert "valid" in datasets
#     assert len(datasets["valid"]) == 1
#     report = datasets["valid"][0]
#     assert "dataset" in report
#     assert "status" in report
#     assert "error" in report


def test_get_valid_datasets(client: TestClient) -> None:
    response = client.get("/valid")
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json


def test_get_valid__next_datasets(client: TestClient) -> None:
    response = client.get("/valid-next")
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json


@responses.activate
def test_get_is_valid(client: TestClient) -> None:
    response = client.get("/is-valid")
    assert response.status_code == 422

    dataset = "doesnotexist"
    responses.add_callback(responses.GET, external_auth_url % dataset, callback=request_callback)
    response = client.get("/is-valid", params={"dataset": dataset})
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json
    assert json["valid"] is False


@responses.activate
def test_get_is_valid_next(client: TestClient) -> None:
    response = client.get("/is-valid-next")
    assert response.status_code == 422

    dataset = "doesnotexist"
    responses.add_callback(responses.GET, external_auth_url % dataset, callback=request_callback)
    response = client.get("/is-valid-next", params={"dataset": dataset})
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json
    assert json["valid"] is False


# the logic below is just to check the cookie and authorization headers are managed correctly
@pytest.mark.parametrize(
    "headers,status_code,error_code",
    [
        ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
        ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
        ({}, 200, None),
    ],
)
@responses.activate
def test_is_valid_auth(
    client: TestClient, headers: Dict[str, str], status_code: int, error_code: Optional[str]
) -> None:
    dataset = "dataset-which-does-not-exist"
    responses.add_callback(responses.GET, external_auth_url % dataset, callback=request_callback)
    response = client.get(f"/is-valid-next?dataset={dataset}", headers=headers)
    assert response.status_code == status_code
    assert response.headers.get("X-Error-Code") == error_code


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_get_splits(client: TestClient) -> None:
    # missing parameter
    response = client.get("/splits")
    assert response.status_code == 400


def test_get_splits_next(client: TestClient) -> None:
    # missing parameter
    response = client.get("/splits-next")
    assert response.status_code == 422
    # empty parameter
    response = client.get("/splits-next?dataset=")
    assert response.status_code == 422


# the logic below is just to check the cookie and authorization headers are managed correctly
@pytest.mark.parametrize(
    "headers,status_code,error_code",
    [
        ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
        ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
        ({}, 404, "SplitsResponseNotFound"),
    ],
)
@responses.activate
def test_splits_next_auth(client: TestClient, headers: Dict[str, str], status_code: int, error_code: str) -> None:
    dataset = "dataset-which-does-not-exist"
    responses.add_callback(responses.GET, external_auth_url % dataset, callback=request_callback)
    response = client.get(f"/splits-next?dataset={dataset}", headers=headers)
    assert response.status_code == status_code
    assert response.headers.get("X-Error-Code") == error_code


def test_get_first_rows(client: TestClient) -> None:
    # missing parameter
    response = client.get("/first-rows")
    assert response.status_code == 422
    response = client.get("/first-rows?dataset=a")
    assert response.status_code == 422
    response = client.get("/first-rows?dataset=a&config=b")
    assert response.status_code == 422
    # empty parameter
    response = client.get("/first-rows?dataset=a&config=b&split=")
    assert response.status_code == 422


def test_get_rows(client: TestClient) -> None:
    response = client.get("/rows")
    assert response.status_code == 400

    # not found
    response = client.get("/rows", params={"dataset": "doesnotexist", "config": "default", "split": "doesnotexist"})
    assert response.status_code == 400


def test_dataset_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    response = client.get("/splits", params={"dataset": dataset})
    assert response.json()["message"] == "The dataset does not exist."
    add_dataset_job(dataset)
    create_or_mark_dataset_as_stale(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.json()["message"] == "The dataset is being processed. Retry later."


def test_split_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.json()["message"] == "The split does not exist."
    add_split_job(dataset, config, split)
    create_or_mark_split_as_stale({"dataset_name": dataset, "config_name": config, "split_name": split}, 0)
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.json()["message"] == "The split is being processed. Retry later."


@responses.activate
def test_splits_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    responses.add_callback(responses.GET, external_auth_url % dataset, callback=request_callback)

    response = client.get("/splits-next", params={"dataset": dataset})
    assert response.json()["error"] == "Not found."
    add_splits_job(dataset)
    mark_splits_responses_as_stale(dataset)
    # ^ has no effect for the moment (no entry for the dataset, and anyway: no way to know the value of the stale flag)
    response = client.get("/splits-next", params={"dataset": dataset})
    assert response.json()["error"] == "The list of splits is not ready yet. Please retry later."
    # simulate the worker
    upsert_splits_response(dataset, {"key": "value"}, HTTPStatus.OK)
    response = client.get("/splits-next", params={"dataset": dataset})
    assert response.json()["key"] == "value"
    assert response.status_code == 200


@responses.activate
def test_first_rows_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    responses.add_callback(responses.GET, external_auth_url % dataset, callback=request_callback)

    response = client.get("/first-rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.json()["error"] == "Not found."
    add_first_rows_job(dataset, config, split)
    mark_first_rows_responses_as_stale(dataset, config, split)
    # ^ has no effect for the moment (no entry for the split, and anyway: no way to know the value of the stale flag)
    response = client.get("/first-rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.json()["error"] == "The list of the first rows is not ready yet. Please retry later."
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
    name = "process_start_time_seconds"
    assert name in metrics
    assert metrics[name] > 0
    name = "process_start_time_seconds"
    assert 'starlette_requests_total{method="GET",path_template="/metrics"}' in metrics
