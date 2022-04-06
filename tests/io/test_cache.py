import pytest
from mongoengine import DoesNotExist

from datasets_preview_backend.config import MONGO_CACHE_DATABASE, ROWS_MAX_NUMBER
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.io.cache import (
    DbDataset,
    clean_database,
    connect_to_cache,
    delete_dataset_cache,
    get_rows_response,
    get_splits_response,
    refresh_dataset_split_full_names,
    refresh_split,
    upsert_dataset,
    upsert_split,
)
from datasets_preview_backend.models.dataset import get_dataset_split_full_names


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise Exception("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache()


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_database()


def test_save() -> None:
    dataset_cache = DbDataset(dataset_name="test", status="valid")
    dataset_cache.save()

    retrieved = DbDataset.objects(dataset_name="test")
    assert len(list(retrieved)) == 1


def test_save_and_update() -> None:
    DbDataset(dataset_name="test", status="empty").save()
    DbDataset.objects(dataset_name="test").upsert_one(status="valid")
    retrieved = DbDataset.objects(dataset_name="test")
    assert len(retrieved) == 1
    assert retrieved[0].status.value == "valid"


def test_acronym_identification() -> None:
    dataset_name = "acronym_identification"
    split_full_names = get_dataset_split_full_names(dataset_name)
    upsert_dataset(dataset_name, split_full_names)
    # ensure it's idempotent
    upsert_dataset(dataset_name, split_full_names)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.dataset_name == dataset_name
    assert retrieved.status.value == "valid"
    delete_dataset_cache(dataset_name)
    with pytest.raises(DoesNotExist):
        DbDataset.objects(dataset_name=dataset_name).get()


def test_doesnotexist() -> None:
    dataset_name = "doesnotexist"
    with pytest.raises(Status400Error):
        refresh_dataset_split_full_names(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status.value == "error"


def test_config_error() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/78
    dataset_name = "Check/region_1"
    refresh_dataset_split_full_names(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status.value == "valid"
    splits_response, error, status_code = get_splits_response(dataset_name)
    assert status_code == 200
    assert error is None
    assert splits_response is not None
    assert "splits" in splits_response
    assert len(splits_response["splits"]) == 1


def test_large_document() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/89
    dataset_name = "SaulLu/Natural_Questions_HTML"
    refresh_dataset_split_full_names(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status.value == "valid"


def test_column_order() -> None:
    refresh_split("acronym_identification", "default", "train")
    rows_response, error, status_code = get_rows_response("acronym_identification", "default", "train")
    assert status_code == 200
    assert error is None
    assert rows_response is not None
    print(rows_response["columns"])
    assert "columns" in rows_response
    assert rows_response["columns"][0]["column"]["name"] == "id"
    assert rows_response["columns"][1]["column"]["name"] == "tokens"
    assert rows_response["columns"][2]["column"]["name"] == "labels"


def test_big_row() -> None:
    # https://github.com/huggingface/datasets-preview-backend/issues/197
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    big_row = {"col": "a" * 100_000_000}
    split = {"split_name": split_name, "rows": [big_row], "columns": [], "num_bytes": None, "num_examples": None}
    upsert_split(dataset_name, config_name, split_name, split)
    rows_response, error, status_code = get_rows_response(dataset_name, config_name, split_name)
    assert status_code == 200
    assert error is None
    assert rows_response["rows"][0]["row"]["col"] == ""
    assert rows_response["rows"][0]["truncated_cells"] == ["col"]
