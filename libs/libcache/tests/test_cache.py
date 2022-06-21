from typing import List

import pytest
from libutils.exceptions import Status400Error
from libutils.types import RowItem, Split, SplitFullName
from mongoengine import DoesNotExist

from libcache.cache import (
    DbDataset,
    DbSplit,
    Status,
    clean_database,
    connect_to_cache,
    delete_dataset_cache,
    get_datasets_count_by_status,
    get_rows_response,
    get_splits_count_by_status,
    get_valid_or_stalled_dataset_names,
    upsert_dataset,
    upsert_split,
    upsert_split_error,
)

from ._utils import MONGO_CACHE_DATABASE, MONGO_URL


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)


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


def test_upsert_dataset() -> None:
    dataset_name = "test_dataset"
    split_full_names: List[SplitFullName] = [
        {"dataset_name": dataset_name, "config_name": "test_config", "split_name": "test_split"}
    ]
    upsert_dataset(dataset_name, split_full_names)
    split = DbSplit.objects(dataset_name=dataset_name).get()
    assert split.status == Status.EMPTY
    # ensure it's idempotent
    upsert_dataset(dataset_name, split_full_names)
    split = DbSplit.objects(dataset_name=dataset_name).get()
    assert split.status == Status.EMPTY
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.dataset_name == dataset_name
    assert retrieved.status.value == "valid"
    delete_dataset_cache(dataset_name)
    with pytest.raises(DoesNotExist):
        DbDataset.objects(dataset_name=dataset_name).get()


def test_big_row() -> None:
    # https://github.com/huggingface/datasets-server/issues/197
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    big_row: RowItem = {
        "dataset": dataset_name,
        "config": config_name,
        "split": split_name,
        "row_idx": 0,
        "row": {"col": "a" * 100_000_000},
        "truncated_cells": [],
    }
    split: Split = {
        "split_name": split_name,
        "rows_response": {"rows": [big_row], "columns": []},
        "num_bytes": None,
        "num_examples": None,
    }
    upsert_split(dataset_name, config_name, split_name, split)
    rows_response, error, status_code = get_rows_response(dataset_name, config_name, split_name)
    assert status_code == 500
    assert error is not None
    assert rows_response is None
    assert error["message"] == "could not store the rows/ cache entry."
    assert error["cause_exception"] == "DocumentTooLarge"


def test_valid() -> None:
    assert get_valid_or_stalled_dataset_names() == []

    upsert_dataset(
        "test_dataset", [{"dataset_name": "test_dataset", "config_name": "test_config", "split_name": "test_split"}]
    )

    assert get_valid_or_stalled_dataset_names() == []

    upsert_split(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "split_name": "test_split",
            "rows_response": {"rows": [], "columns": []},
            "num_bytes": None,
            "num_examples": None,
        },
    )

    assert get_valid_or_stalled_dataset_names() == ["test_dataset"]

    upsert_dataset(
        "test_dataset2",
        [
            {"dataset_name": "test_dataset2", "config_name": "test_config2", "split_name": "test_split2"},
            {"dataset_name": "test_dataset2", "config_name": "test_config2", "split_name": "test_split3"},
        ],
    )

    assert get_valid_or_stalled_dataset_names() == ["test_dataset"]

    upsert_split_error("test_dataset2", "test_config2", "test_split2", Status400Error("error"))

    assert get_valid_or_stalled_dataset_names() == ["test_dataset"]

    upsert_split(
        "test_dataset2",
        "test_config2",
        "test_split3",
        {
            "split_name": "test_split3",
            "rows_response": {"rows": [], "columns": []},
            "num_bytes": None,
            "num_examples": None,
        },
    )

    assert get_valid_or_stalled_dataset_names() == ["test_dataset", "test_dataset2"]


def test_count_by_status() -> None:
    assert get_datasets_count_by_status() == {"empty": 0, "error": 0, "stalled": 0, "valid": 0}

    upsert_dataset(
        "test_dataset", [{"dataset_name": "test_dataset", "config_name": "test_config", "split_name": "test_split"}]
    )

    assert get_datasets_count_by_status() == {"empty": 0, "error": 0, "stalled": 0, "valid": 1}
    assert get_splits_count_by_status() == {"empty": 1, "error": 0, "stalled": 0, "valid": 0}

    upsert_split(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "split_name": "test_split",
            "rows_response": {"rows": [], "columns": []},
            "num_bytes": None,
            "num_examples": None,
        },
    )

    assert get_splits_count_by_status() == {"empty": 0, "error": 0, "stalled": 0, "valid": 1}
