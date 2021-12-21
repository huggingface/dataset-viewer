import pytest
from starlette.testclient import TestClient

from datasets_preview_backend.app import create_app
from datasets_preview_backend.config import MONGO_CACHE_DATABASE
from datasets_preview_backend.io.cache import clean_database, refresh_dataset


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise Exception("Test must be launched on a test mongo database")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_get_cache_reports(client: TestClient) -> None:
    refresh_dataset("acronym_identification")
    response = client.get("/cache-reports")
    assert response.status_code == 200
    json = response.json()
    reports = json["reports"]
    assert len(reports) == 1
    report = reports[0]
    assert "dataset" in report
    assert "status" in report
    assert "error" in report


def test_get_cache_stats(client: TestClient) -> None:
    response = client.get("/cache")
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json
    assert "error" in json


def test_get_valid_datasets(client: TestClient) -> None:
    response = client.get("/valid")
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_get_hf_datasets(client: TestClient) -> None:
    response = client.get("/hf_datasets")
    assert response.status_code == 200
    json = response.json()
    datasets = json["datasets"]
    assert len(datasets) > 1000


def test_get_splits(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    refresh_dataset(dataset)
    response = client.get("/splits", params={"dataset": dataset, "config": config})
    assert response.status_code == 200
    json = response.json()
    splitItems = json["splits"]
    assert len(splitItems) == 3
    split = splitItems[0]
    assert split["dataset"] == dataset
    assert split["config"] == config
    assert split["split"] == "test"

    # no config
    response2 = client.get("/splits", params={"dataset": dataset})
    json2 = response2.json()
    assert json == json2

    # not found
    response = client.get("/splits", params={"dataset": dataset, "config": "doesnotexist"})
    assert response.status_code == 404

    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    dataset = "hda_nli_hindi"
    config = "HDA nli hindi"
    refresh_dataset(dataset)
    response = client.get("/splits", params={"dataset": dataset, "config": config})
    assert response.status_code == 200
    json = response.json()
    splits = [s["split"] for s in json["splits"]]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    # not found
    dataset = "doesnotexist"
    refresh_dataset(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.status_code == 404


def test_get_rows(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    refresh_dataset(dataset)
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 200
    json = response.json()
    rowItems = json["rows"]
    assert len(rowItems) > 3
    rowItem = rowItems[0]
    assert rowItem["dataset"] == dataset
    assert rowItem["config"] == config
    assert rowItem["split"] == split
    assert rowItem["row"]["tokens"][0] == "What"

    assert len(json["columns"]) == 3
    column_item = json["columns"][0]
    assert "dataset" in column_item
    assert "config" in column_item
    assert "column_idx" in column_item
    column = column_item["column"]
    assert column["name"] == "id"
    assert column["type"] == "STRING"

    # no split
    response2 = client.get("/rows", params={"dataset": dataset, "config": config})
    json2 = response2.json()
    assert len(json2["rows"]) > len(json["rows"])

    # no config
    response3 = client.get("/rows", params={"dataset": dataset})
    json3 = response3.json()
    assert json2 == json3

    # not found
    response = client.get("/rows", params={"dataset": dataset, "config": "doesnotexist"})
    assert response.status_code == 404

    response = client.get("/rows", params={"dataset": dataset, "config": "default", "split": "doesnotexist"})
    assert response.status_code == 404

    # not found
    dataset = "doesnotexist"
    refresh_dataset(dataset)
    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 404


def test_datetime_content(client: TestClient) -> None:
    dataset = "allenai/c4"
    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 404

    refresh_dataset(dataset)

    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 200
