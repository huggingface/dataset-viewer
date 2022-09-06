from typing import Optional

import pytest

# from libcache.cache import clean_database as clean_cache_database
from libcache.cache import clean_database as clean_cache_database
from libqueue.queue import clean_database as clean_queue_database
from starlette.testclient import TestClient

from admin.app import create_app
from admin.config import MONGO_CACHE_DATABASE, MONGO_QUEUE_DATABASE


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Tests on cache must be launched on a test mongo database")
    if "test" not in MONGO_QUEUE_DATABASE:
        raise ValueError("Tests on queue must be launched on a test mongo database")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_databases() -> None:
    clean_cache_database()
    clean_queue_database()


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
    name = "process_start_time_seconds"
    assert name in metrics
    assert metrics[name] > 0
    name = "process_start_time_seconds"
    assert 'queue_jobs_total{queue="/splits",status="waiting"}' in metrics
    assert 'queue_jobs_total{queue="/rows",status="success"}' in metrics
    assert 'queue_jobs_total{queue="/splits-next",status="started"}' in metrics
    assert 'queue_jobs_total{queue="/first-rows",status="started"}' in metrics
    assert 'cache_entries_total{cache="/splits",status="valid"}' in metrics
    # still empty
    assert 'responses_in_cache_total{path="/rows",http_status="200",error_code=null}' not in metrics
    # still empty
    assert 'responses_in_cache_total{path="/splits-next",http_status="200",error_code=null}' not in metrics
    assert 'responses_in_cache_total{path="/first-rows",http_status="200",error_code=null}' not in metrics
    assert 'starlette_requests_total{method="GET",path_template="/metrics"}' in metrics


def test_pending_jobs(client: TestClient) -> None:
    response = client.get("/pending-jobs")
    assert response.status_code == 200
    json = response.json()
    for e in ["/splits", "/rows", "/splits-next", "/first-rows"]:
        assert json[e] == {"waiting": [], "started": []}
    assert "created_at" in json


@pytest.mark.parametrize(
    "path,cursor,http_status,error_code",
    [
        ("/splits-next", None, 200, None),
        ("/splits-next", "", 200, None),
        ("/splits-next", "invalid cursor", 422, "InvalidParameter"),
        ("/first-rows", None, 200, None),
        ("/first-rows", "", 200, None),
        ("/first-rows", "invalid cursor", 422, "InvalidParameter"),
    ],
)
def test_cache_reports(
    client: TestClient, path: str, cursor: Optional[str], http_status: int, error_code: Optional[str]
) -> None:
    cursor_str = f"?cursor={cursor}" if cursor else ""
    response = client.get(f"/cache-reports{path}{cursor_str}")
    assert response.status_code == http_status
    if error_code:
        assert isinstance(response.json()["error"], str)
        assert response.headers["X-Error-Code"] == error_code
    else:
        assert response.json() == {"cache_reports": [], "next_cursor": ""}
        assert "X-Error-Code" not in response.headers
