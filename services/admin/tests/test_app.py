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
    assert 'queue_jobs_total{queue="datasets",status="waiting"}' in metrics
    # assert 'cache_entries_total{cache="datasets",status="empty"}' in metrics
    assert 'starlette_requests_total{method="GET",path_template="/metrics"}' in metrics
