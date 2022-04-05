import pytest
from starlette.testclient import TestClient

from datasets_preview_backend.app import create_app
from datasets_preview_backend.config import MONGO_CACHE_DATABASE, MONGO_QUEUE_DATABASE
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.io.cache import clean_database as clean_cache_database
from datasets_preview_backend.io.cache import (
    create_or_mark_dataset_as_stalled,
    create_or_mark_split_as_stalled,
    refresh_dataset_split_full_names,
    refresh_split,
)
from datasets_preview_backend.io.queue import add_dataset_job, add_split_job
from datasets_preview_backend.io.queue import clean_database as clean_queue_database
from datasets_preview_backend.io.queue import (
    finish_dataset_job,
    finish_split_job,
    get_dataset_job,
)


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise Exception("Tests on cache must be launched on a test mongo database")
    if "test" not in MONGO_QUEUE_DATABASE:
        raise Exception("Tests on queue must be launched on a test mongo database")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_databases() -> None:
    clean_cache_database()
    clean_queue_database()


def test_get_cache_reports(client: TestClient) -> None:
    refresh_dataset_split_full_names("acronym_identification")
    response = client.get("/cache-reports")
    assert response.status_code == 200
    json = response.json()
    assert "datasets" in json
    assert "splits" in json
    datasets = json["datasets"]
    assert "empty" in datasets
    assert "error" in datasets
    assert "stalled" in datasets
    assert "valid" in datasets
    assert len(datasets["valid"]) == 1
    report = datasets["valid"][0]
    assert "dataset" in report
    assert "status" in report
    assert "error" in report


def test_get_cache_stats(client: TestClient) -> None:
    response = client.get("/cache")
    assert response.status_code == 200
    json = response.json()
    assert "datasets" in json
    assert "splits" in json
    datasets = json["datasets"]
    assert "empty" in datasets
    assert "error" in datasets
    assert "stalled" in datasets
    assert "valid" in datasets


def test_get_valid_datasets(client: TestClient) -> None:
    response = client.get("/valid")
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json


def test_get_is_valid(client: TestClient) -> None:
    response = client.get("/is-valid")
    assert response.status_code == 400

    dataset = "acronym_identification"
    split_full_names = refresh_dataset_split_full_names(dataset)
    for split_full_name in split_full_names:
        refresh_split(split_full_name["dataset_name"], split_full_name["config_name"], split_full_name["split_name"])
    response = client.get("/is-valid", params={"dataset": "acronym_identification"})
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json
    assert json["valid"] is True

    response = client.get("/is-valid", params={"dataset": "doesnotexist"})
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json
    assert json["valid"] is False


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
    refresh_dataset_split_full_names(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.status_code == 200
    json = response.json()
    splitItems = json["splits"]
    assert len(splitItems) == 3
    split = splitItems[0]
    assert split["dataset"] == dataset
    assert split["config"] == "default"
    assert split["split"] == "train"

    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    dataset = "hda_nli_hindi"
    refresh_dataset_split_full_names(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.status_code == 200
    json = response.json()
    splits = [s["split"] for s in json["splits"]]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    # not found
    dataset = "doesnotexist"
    with pytest.raises(Status400Error):
        refresh_dataset_split_full_names(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.status_code == 400

    # missing parameter
    response = client.get("/splits")
    assert response.status_code == 400


def test_get_rows(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    refresh_split(dataset, config, split)
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

    # missing parameter
    response = client.get("/rows", params={"dataset": dataset, "config": config})
    assert response.status_code == 400
    response = client.get("/rows", params={"dataset": dataset})
    assert response.status_code == 400
    response = client.get("/rows")
    assert response.status_code == 400

    # not found
    response = client.get("/rows", params={"dataset": dataset, "config": "default", "split": "doesnotexist"})
    assert response.status_code == 400


def test_datetime_content(client: TestClient) -> None:
    dataset = "allenai/c4"
    config = "allenai--c4"
    split = "train"
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 400

    refresh_split(dataset, config, split)

    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 200


def test_bytes_limit(client: TestClient) -> None:
    dataset = "edbeeching/decision_transformer_gym_replay"
    config = "hopper-expert-v2"
    split = "train"
    refresh_split(dataset, config, split)
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 200
    json = response.json()
    rowItems = json["rows"]
    assert len(rowItems) == 3


def test_dataset_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    response = client.get("/splits", params={"dataset": dataset})
    assert response.json()["message"] == "The dataset does not exist."
    add_dataset_job(dataset)
    create_or_mark_dataset_as_stalled(dataset)
    response = client.get("/splits", params={"dataset": dataset})
    assert response.json()["message"] == "The dataset is being processed. Retry later."


def test_split_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.json()["message"] == "The split does not exist."
    add_split_job(dataset, config, split)
    create_or_mark_split_as_stalled({"dataset_name": dataset, "config_name": config, "split_name": split}, 0)
    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    assert response.json()["message"] == "The split is being processed. Retry later."


def test_error_messages(client: TestClient) -> None:
    # https://github.com/huggingface/datasets-preview-backend/issues/196
    dataset = "acronym_identification"
    config = "default"
    split = "train"

    response = client.get("/splits", params={"dataset": dataset})
    # ^ equivalent to
    # curl http://localhost:8000/splits\?dataset\=acronym_identification
    assert response.json()["message"] == "The dataset does not exist."

    client.post("/webhook", json={"update": f"datasets/{dataset}"})
    # ^ equivalent to
    # curl -X POST http://localhost:8000/webhook -H 'Content-Type: application/json' \
    #   -d '{"update": "datasets/acronym_identification"}'

    response = client.get("/splits", params={"dataset": dataset})
    # ^ equivalent to
    # curl http://localhost:8000/splits\?dataset\=acronym_identification
    assert response.json()["message"] == "The dataset is being processed. Retry later."

    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    # ^ equivalent to
    # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train
    assert response.json()["message"] == "The dataset is being processed. Retry later."

    # simulate dataset worker
    # ^ equivalent to
    # WORKER_QUEUE=datasets make worker
    # part A
    job_id, dataset_name = get_dataset_job()
    split_full_names = refresh_dataset_split_full_names(dataset_name=dataset_name)

    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    # ^ equivalent to
    # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train
    assert response.status_code == 500
    assert response.json()["message"] == "The split cache is empty but no job has been launched."

    # part B
    for split_full_name in split_full_names:
        add_split_job(split_full_name["dataset_name"], split_full_name["config_name"], split_full_name["split_name"])
    finish_dataset_job(job_id, success=True)

    response = client.get("/splits", params={"dataset": dataset})
    # ^ equivalent to
    # curl http://localhost:8000/splits\?dataset\=acronym_identification
    assert response.status_code == 200
    assert response.json()["splits"][0] == {
        "dataset": dataset,
        "config": config,
        "split": split,
        "num_bytes": None,
        "num_examples": None,
    }

    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    # ^ equivalent to
    # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train
    assert response.json()["message"] == "The split is being processed. Retry later."

    refresh_split(dataset_name=dataset, config_name=config, split_name=split)
    finish_split_job(job_id, success=True)
    # ^ equivalent to
    # WORKER_QUEUE=splits make worker

    response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    # ^ equivalent to
    # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train

    assert response.status_code == 200
    assert len(response.json()["rows"]) > 0
