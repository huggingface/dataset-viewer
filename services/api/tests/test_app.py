import pytest

# from libcache.cache import clean_database as clean_cache_database
from libcache.cache import clean_database as clean_cache_database
from libcache.cache import (
    create_or_mark_dataset_as_stale,
    create_or_mark_split_as_stale,
)
from libcache.simple_cache import (
    HTTPStatus,
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
from api.config import MONGO_QUEUE_DATABASE


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


def test_get_is_valid(client: TestClient) -> None:
    response = client.get("/is-valid")
    assert response.status_code == 400

    response = client.get("/is-valid", params={"dataset": "doesnotexist"})
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json
    assert json["valid"] is False

    # TODO: move to e2e tests
    #     dataset = "acronym_identification"
    #     split_full_names = refresh_dataset_split_full_names(dataset)
    #     for split_full_name in split_full_names:
    #         refresh_split(
    #             split_full_name["dataset_name"],
    #             split_full_name["config_name"],
    #             split_full_name["split_name"],
    #             rows_max_bytes=ROWS_MAX_BYTES,
    #             rows_max_number=ROWS_MAX_NUMBER,
    #             rows_min_number=ROWS_MIN_NUMBER,
    #         )
    #     response = client.get("/is-valid", params={"dataset": "acronym_identification"})
    #     assert response.status_code == 200
    #     json = response.json()
    #     assert "valid" in json
    #     assert json["valid"] is True


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_get_splits(client: TestClient) -> None:
    # TODO: move to e2e tests
    #     dataset = "acronym_identification"
    #     refresh_dataset_split_full_names(dataset)
    #     response = client.get("/splits", params={"dataset": dataset})
    #     assert response.status_code == 200
    #     json = response.json()
    #     splitItems = json["splits"]
    #     assert len(splitItems) == 3
    #     split = splitItems[0]
    #     assert split["dataset"] == dataset
    #     assert split["config"] == "default"
    #     assert split["split"] == "train"

    #     # uses the fallback to call "builder._split_generators"
    #     # while https://github.com/huggingface/datasets/issues/2743
    #     dataset = "hda_nli_hindi"
    #     refresh_dataset_split_full_names(dataset)
    #     response = client.get("/splits", params={"dataset": dataset})
    #     assert response.status_code == 200
    #     json = response.json()
    #     splits = [s["split"] for s in json["splits"]]
    #     assert len(splits) == 3
    #     assert "train" in splits
    #     assert "validation" in splits
    #     assert "test" in splits

    #     # not found
    #     dataset = "doesnotexist"
    #     with pytest.raises(Status400Error):
    #         refresh_dataset_split_full_names(dataset)
    #     response = client.get("/splits", params={"dataset": dataset})
    #     assert response.status_code == 400

    # missing parameter
    response = client.get("/splits")
    assert response.status_code == 400


def test_get_rows(client: TestClient) -> None:
    # TODO: move to e2e tests
    # # dataset = "acronym_identification"
    # # config = "default"
    # # split = "train"
    # # refresh_split(
    # #     dataset,
    # #     config,
    # #     split,
    # #     rows_max_bytes=ROWS_MAX_BYTES,
    # #     rows_max_number=ROWS_MAX_NUMBER,
    # #     rows_min_number=ROWS_MIN_NUMBER,
    # # )
    # # response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
    # # assert response.status_code == 200
    # # json = response.json()
    # # rowItems = json["rows"]
    # # assert len(rowItems) > 3
    # # rowItem = rowItems[0]
    # # assert rowItem["dataset"] == dataset
    # # assert rowItem["config"] == config
    # # assert rowItem["split"] == split
    # # assert rowItem["row"]["tokens"][0] == "What"

    # # assert len(json["columns"]) == 3
    # # column_item = json["columns"][0]
    # # assert "dataset" in column_item
    # # assert "config" in column_item
    # # assert "column_idx" in column_item
    # # column = column_item["column"]
    # # assert column["name"] == "id"
    # # assert column["type"] == "STRING"

    # missing parameter
    # response = client.get("/rows", params={"dataset": dataset, "config": config})
    # assert response.status_code == 400
    # response = client.get("/rows", params={"dataset": dataset})
    # assert response.status_code == 400
    response = client.get("/rows")
    assert response.status_code == 400

    # not found
    response = client.get("/rows", params={"dataset": "doesnotexist", "config": "default", "split": "doesnotexist"})
    assert response.status_code == 400


# TODO: move to e2e tests
# def test_datetime_content(client: TestClient) -> None:
#     dataset = "allenai/c4"
#     config = "allenai--c4"
#     split = "train"
#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     assert response.status_code == 400

#     refresh_split(
#         dataset,
#         config,
#         split,
#         rows_max_bytes=ROWS_MAX_BYTES,
#         rows_max_number=ROWS_MAX_NUMBER,
#         rows_min_number=ROWS_MIN_NUMBER,
#     )

#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     assert response.status_code == 200

# TODO: move to e2e tests
# def test_bytes_limit(client: TestClient) -> None:
#     dataset = "edbeeching/decision_transformer_gym_replay"
#     config = "hopper-expert-v2"
#     split = "train"
#     refresh_split(
#         dataset,
#         config,
#         split,
#         rows_max_bytes=ROWS_MAX_BYTES,
#         rows_max_number=ROWS_MAX_NUMBER,
#         rows_min_number=ROWS_MIN_NUMBER,
#     )
#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     assert response.status_code == 200
#     # json = response.json()
#     # rowItems = json["rows"]
#     # assert len(rowItems) == 3
#     # TODO: re-enable and fix the test after the refactoring


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


def test_splits_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
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


def test_first_rows_cache_refreshing(client: TestClient) -> None:
    dataset = "acronym_identification"
    config = "default"
    split = "train"
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


# def test_split_cache_refreshing(client: TestClient) -> None:
#     dataset = "acronym_identification"
#     config = "default"
#     split = "train"
#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     assert response.json()["message"] == "The split does not exist."
#     add_split_job(dataset, config, split)
#     create_or_mark_split_as_stale({"dataset_name": dataset, "config_name": config, "split_name": split}, 0)
#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     assert response.json()["message"] == "The split is being processed. Retry later."


# TODO: move to e2e tests
# def test_error_messages(client: TestClient) -> None:
#     # https://github.com/huggingface/datasets-server/issues/196
#     dataset = "acronym_identification"
#     config = "default"
#     split = "train"

#     response = client.get("/splits", params={"dataset": dataset})
#     # ^ equivalent to
#     # curl http://localhost:8000/splits\?dataset\=acronym_identification
#     assert response.json()["message"] == "The dataset does not exist."

#     client.post("/webhook", json={"update": f"datasets/{dataset}"})
#     # ^ equivalent to
#     # curl -X POST http://localhost:8000/webhook -H 'Content-Type: application/json' \
#     #   -d '{"update": "datasets/acronym_identification"}'

#     response = client.get("/splits", params={"dataset": dataset})
#     # ^ equivalent to
#     # curl http://localhost:8000/splits\?dataset\=acronym_identification
#     assert response.json()["message"] == "The dataset is being processed. Retry later."

#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     # ^ equivalent to
#     # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train
#     assert response.json()["message"] == "The dataset is being processed. Retry later."

#     # simulate dataset worker
#     # ^ equivalent to
#     # WORKER_QUEUE=datasets make worker
#     # part A
#     job_id, dataset_name = get_dataset_job()
#     split_full_names = refresh_dataset_split_full_names(dataset_name=dataset_name)

#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     # ^ equivalent to
#     # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train
#     assert response.status_code == 500
#     assert response.json()["message"] == "The split cache is empty but no job has been launched."

#     # part B
#     for split_full_name in split_full_names:
#         add_split_job(split_full_name["dataset_name"], split_full_name["config_name"], split_full_name["split_name"])
#     finish_dataset_job(job_id, success=True)

#     response = client.get("/splits", params={"dataset": dataset})
#     # ^ equivalent to
#     # curl http://localhost:8000/splits\?dataset\=acronym_identification
#     assert response.status_code == 200
#     assert response.json()["splits"][0] == {
#         "dataset": dataset,
#         "config": config,
#         "split": split,
#         "num_bytes": None,
#         "num_examples": None,
#     }

#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     # ^ equivalent to
#     # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train
#     assert response.json()["message"] == "The split is being processed. Retry later."

#     refresh_split(
#         dataset_name=dataset,
#         config_name=config,
#         split_name=split,
#         rows_max_bytes=ROWS_MAX_BYTES,
#         rows_max_number=ROWS_MAX_NUMBER,
#         rows_min_number=ROWS_MIN_NUMBER,
#     )
#     finish_split_job(job_id, success=True)
#     # ^ equivalent to
#     # WORKER_QUEUE=splits make worker

#     response = client.get("/rows", params={"dataset": dataset, "config": config, "split": split})
#     # ^ equivalent to
#     # curl http://localhost:8000/rows\?dataset\=acronym_identification\&config\=default\&split\=train

#     assert response.status_code == 200
#     assert len(response.json()["rows"]) > 0


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
    # Disable for now - see https://github.com/huggingface/datasets-server/issues/250#issuecomment-1135561566
    # assert 'queue_jobs_total{queue="datasets",status="waiting"}' in metrics
    # assert 'cache_entries_total{cache="datasets",status="empty"}' in metrics
    assert 'starlette_requests_total{method="GET",path_template="/metrics"}' in metrics
