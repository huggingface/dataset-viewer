# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240626151600_cache_remove_has_fts_field_in_split_duckdb_index import (
    MigrationRemoveHasFTSFromSplitDuckdbIndexCacheResponse,
)


def assert_with_has_fts_field(dataset: str, kind: str, value: bool) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["has_fts"] == value


def assert_without_has_fts_fiel(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "has_fts" not in entry["content"]


def test_cache_remove_has_fts(mongo_host: str) -> None:
    with MongoResource(database="test_cache_remove_has_fts", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        cache: list[dict[str, Any]] = [
            {
                "config": "default",
                "dataset": "dataset_with_fts",
                "kind": "split-duckdb-index",
                "split": "train",
                "content": {
                    "dataset": "dataset_with_fts",
                    "config": "default",
                    "split": "train",
                    "url": "https://huggingface.co/.../index.duckdb",
                    "filename": "index.duckdb",
                    "size": 5038,
                    "has_fts": True,
                    "stemmer": "porter",
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": None,
            },
            {
                "config": "default",
                "dataset": "dataset_without_fts",
                "kind": "split-duckdb-index",
                "split": "train",
                "content": {
                    "dataset": "dataset_without_fts",
                    "config": "default",
                    "split": "train",
                    "url": "https://huggingface.co/.../index.duckdb",
                    "filename": "index.duckdb",
                    "size": 5038,
                    "has_fts": False,
                    "stemmer": None,
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

        migration = MigrationRemoveHasFTSFromSplitDuckdbIndexCacheResponse(
            version="20240626151600",
            description="remove 'has_fts' field from split-duckdb-index",
        )
        migration.up()

        assert_without_has_fts_fiel("dataset_with_fts", kind="split-duckdb-index")
        assert_without_has_fts_fiel("dataset_without_fts", kind="split-duckdb-index")
        assert_without_has_fts_fiel("dataset_with_error", kind="split-duckdb-index")

        migration.down()
        assert_with_has_fts_field("dataset_with_fts", kind="split-duckdb-index", value=True)
        assert_with_has_fts_field("dataset_without_fts", kind="split-duckdb-index", value=False)
        assert_without_has_fts_fiel("dataset_with_error", kind="split-duckdb-index")

        db[CACHE_COLLECTION_RESPONSES].drop()
