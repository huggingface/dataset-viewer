# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240624144000_cache_add_estimated_num_rows_field_in_size import (
    MigrationAddEstimatedNumRowsToSizeCacheResponse,
)


def assert_estimated_num_rows_in_config(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["size"]["config"]["estimated_num_rows"] is None
    assert all(split["estimated_num_rows"] is None for split in entry["content"]["size"]["splits"])


def assert_estimated_num_rows_in_dataset(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["size"]["dataset"]["estimated_num_rows"] is None
    assert all(split["estimated_num_rows"] is None for split in entry["content"]["size"]["configs"])
    assert all(split["estimated_num_rows"] is None for split in entry["content"]["size"]["splits"])


def assert_unchanged_in_config(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    if "size" in entry["content"]:
        assert "estimated_num_rows" not in entry["content"]["size"]["config"]
        assert all("estimated_num_rows" not in split for split in entry["content"]["size"]["splits"])


def assert_unchanged_in_dataset(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    if "size" in entry["content"]:
        assert "estimated_num_rows" not in entry["content"]["size"]["dataset"]
        assert all("estimated_num_rows" not in split for split in entry["content"]["size"]["configs"])
        assert all("estimated_num_rows" not in split for split in entry["content"]["size"]["splits"])


def test_cache_add_partial(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_tags_to_hub_cache", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        cache: list[dict[str, Any]] = [
            {
                "dataset": "dataset",
                "config": "config",
                "kind": "config-size",
                "content": {
                    "size": {
                        "config": {
                            "dataset": "dataset",
                            "config": "config",
                            "num_bytes_original_files": 123,
                            "num_bytes_parquet_files": 123,
                            "num_bytes_memory": 123,
                            "num_rows": 1000,
                            "num_columns": 1,
                        },
                        "splits": [
                            {
                                "dataset": "dataset",
                                "config": "config",
                                "split": "train",
                                "num_bytes_original_files": 120,
                                "num_bytes_parquet_files": 120,
                                "num_bytes_memory": 120,
                                "num_rows": 900,
                                "num_columns": 1,
                            },
                            {
                                "dataset": "dataset",
                                "config": "config",
                                "split": "test",
                                "num_bytes_original_files": 3,
                                "num_bytes_parquet_files": 3,
                                "num_bytes_memory": 3,
                                "num_rows": 100,
                                "num_columns": 1,
                            },
                        ],
                    },
                    "partial": False,
                },
                "http_status": 200,
                "job_runner_version": 1,
                "progress": None,
            },
            {
                "dataset": "dataset",
                "config": "config",
                "kind": "dataset-size",
                "content": {
                    "size": {
                        "dataset": {
                            "dataset": "dataset",
                            "config": "config",
                            "num_bytes_original_files": 123,
                            "num_bytes_parquet_files": 123,
                            "num_bytes_memory": 123,
                            "num_rows": 1000,
                            "num_columns": 1,
                        },
                        "configs": [
                            {
                                "dataset": "dataset",
                                "config": "config",
                                "num_bytes_original_files": 123,
                                "num_bytes_parquet_files": 123,
                                "num_bytes_memory": 123,
                                "num_rows": 1000,
                                "num_columns": 1,
                            }
                        ],
                        "splits": [
                            {
                                "dataset": "dataset",
                                "config": "config",
                                "split": "train",
                                "num_bytes_original_files": 120,
                                "num_bytes_parquet_files": 120,
                                "num_bytes_memory": 120,
                                "num_rows": 900,
                                "num_columns": 1,
                            },
                            {
                                "dataset": "dataset",
                                "config": "config",
                                "split": "test",
                                "num_bytes_original_files": 3,
                                "num_bytes_parquet_files": 3,
                                "num_bytes_memory": 3,
                                "num_rows": 100,
                                "num_columns": 1,
                            },
                        ],
                    },
                    "partial": False,
                },
                "http_status": 200,
                "job_runner_version": 1,
                "progress": None,
            },
            {
                "dataset": "dataset_with_error",
                "config": "config_with_error",
                "kind": "config-size",
                "content": {"error": "error"},
                "details": {
                    "error": "error",
                    "cause_exception": "UnexpextedError",
                    "cause_message": "error",
                    "cause_traceback": ["Traceback"],
                },
                "error_code": "UnexpectedError",
                "http_status": 500,
                "job_runner_version": 1,
                "progress": None,
            },
        ]

        db[CACHE_COLLECTION_RESPONSES].insert_many(cache)

        migration = MigrationAddEstimatedNumRowsToSizeCacheResponse(
            version="20240624144000",
            description="add the 'estimated_num_rows' fields to size",
        )
        migration.up()

        assert_estimated_num_rows_in_config("dataset", kind="config-size")
        assert_estimated_num_rows_in_dataset("dataset", kind="dataset-size")
        assert_unchanged_in_config("dataset_with_error", kind="config-size")

        migration.down()
        assert_unchanged_in_config("dataset", kind="config-size")
        assert_unchanged_in_dataset("dataset", kind="dataset-size")
        assert_unchanged_in_config("dataset_with_error", kind="config-size")

        db[CACHE_COLLECTION_RESPONSES].drop()
