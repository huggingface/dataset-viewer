# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20231106193200_cache_add_partial_field_in_split_duckdb_index import (
    MigrationAddPartialToSplitDuckdbIndexCacheResponse,
)


def assert_partial_and_num_rows_and_num_bytes_fields(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["partial"] is None
    assert entry["content"]["num_rows"] is None
    assert entry["content"]["num_bytes"] is None


def assert_unchanged(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "partial" not in entry["content"]
    assert "num_rows" not in entry["content"]
    assert "num_bytes" not in entry["content"]


def test_cache_add_partial(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_partial", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        cache: list[dict[str, Any]] = [
            {
                "config": "default",
                "dataset": "dataset",
                "kind": "split-duckdb-index",
                "split": "train",
                "content": {
                    "dataset": "dataset",
                    "config": "default",
                    "split": "train",
                    "url": "https://huggingface.co/.../index.duckdb",
                    "filename": "index.duckdb",
                    "size": 5038,
                    "has_fts": False,
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": None,
            },
            {
                "config": "default",
                "dataset": "dataset_with_error",
                "kind": "split-duckdb-index",
                "split": "train",
                "content": {"error": "error"},
                "details": {
                    "error": "error",
                    "cause_exception": "UnexpextedError",
                    "cause_message": "error",
                    "cause_traceback": ["Traceback"],
                },
                "error_code": "UnexpectedError",
                "http_status": 500,
                "job_runner_version": 3,
                "progress": None,
            },
        ]

        db[CACHE_COLLECTION_RESPONSES].insert_many(cache)

        migration = MigrationAddPartialToSplitDuckdbIndexCacheResponse(
            version="20231106193200",
            description="add 'partial', 'num_rows' and 'num_bytes' fields to split-duckdb-index",
        )
        migration.up()

        assert_partial_and_num_rows_and_num_bytes_fields("dataset", kind="split-duckdb-index")
        assert_unchanged("dataset_with_error", kind="split-duckdb-index")

        migration.down()
        assert_unchanged("dataset", kind="split-duckdb-index")
        assert_unchanged("dataset_with_error", kind="split-duckdb-index")

        db[CACHE_COLLECTION_RESPONSES].drop()
