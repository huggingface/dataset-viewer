import pytest
from starlette.testclient import TestClient

from datasets_preview_backend.app import create_app
from datasets_preview_backend.config import MONGO_CACHE_DATABASE
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.io.cache import clean_database, upsert_dataset_cache
from datasets_preview_backend.models.dataset import get_dataset


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


def compute_dataset_cache(dataset_name: str) -> None:
    try:
        dataset = get_dataset(dataset_name=dataset_name)
        upsert_dataset_cache(dataset_name, "valid", dataset)
    except StatusError as err:
        upsert_dataset_cache(dataset_name, "error", err.as_content())


def test_get_cache_reports(client: TestClient) -> None:
    compute_dataset_cache("acronym_identification")
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


def test_get_configs(client: TestClient) -> None:
    dataset = "acronym_identification"
    compute_dataset_cache(dataset)
    response = client.get("/configs", params={"dataset": dataset})
    assert response.status_code == 200
    json = response.json()
    configs = json["configs"]
    assert len(configs) == 1
    config = configs[0]
    assert config["dataset"] == dataset
    assert config["config"] == "default"


def test_get_infos(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    compute_dataset_cache(dataset)
    response = client.get("/infos", params={"dataset": dataset, "config": config})
    assert response.status_code == 200
    json = response.json()
    infoItems = json["infos"]
    assert len(infoItems) == 1
    infoItem = infoItems[0]
    assert infoItem["dataset"] == dataset
    assert infoItem["config"] == config
    assert "features" in infoItem["info"]

    # no config provided
    response = client.get("/infos", params={"dataset": dataset})
    json = response.json()
    infoItems = json["infos"]
    assert len(infoItems) == 1

    # not found
    response = client.get("/infos", params={"dataset": dataset, "config": "doesnotexist"})
    assert response.status_code == 404

    # no dataset info file
    dataset = "lhoestq/custom_squad"
    compute_dataset_cache(dataset)
    response = client.get("/infos", params={"dataset": dataset})
    json = response.json()
    infoItems = json["infos"]
    assert len(infoItems) == 1

    # not found
    dataset = "doesnotexist"
    compute_dataset_cache(dataset)
    response = client.get("/infos", params={"dataset": dataset})
    assert response.status_code == 404


def test_get_splits(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    compute_dataset_cache(dataset)
    response = client.get("/splits", params={"dataset": dataset, "config": config})
    assert response.status_code == 200
    json = response.json()
    splitItems = json["splits"]
    assert len(splitItems) == 3
    split = splitItems[0]
    assert split["dataset"] == dataset
    assert split["config"] == config
    assert split["split"] == "train"

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
    compute_dataset_cache(dataset)
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
    compute_dataset_cache(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.status_code == 404


def test_get_rows(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    compute_dataset_cache(dataset)
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
    compute_dataset_cache(dataset)
    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 404


def test_datetime_content(client: TestClient) -> None:
    dataset = "allenai/c4"
    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 404

    compute_dataset_cache(dataset)

    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 200
