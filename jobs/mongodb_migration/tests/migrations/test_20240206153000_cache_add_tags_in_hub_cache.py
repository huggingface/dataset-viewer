# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240206153000_cache_add_tags_in_hub_cache import (
    MigrationAddTagsToHubCacheCacheResponse,
)


def assert_tags(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["tags"] == []


def assert_unchanged(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "tags" not in entry["content"]


def test_cache_add_partial(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_tags_to_hub_cache", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        cache: list[dict[str, Any]] = [
            {
                "dataset": "dataset",
                "kind": "dataset-hub-cache",
                "content": {
                    "viewer": True,
                    "preview": True,
                    "num_rows": 100,
                    "partial": False,
                },
                "http_status": 200,
                "job_runner_version": 1,
                "progress": None,
            },
            {
                "dataset": "dataset_with_error",
                "kind": "dataset-hub-cache",
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

        migration = MigrationAddTagsToHubCacheCacheResponse(
            version="20240206153000",
            description="add the 'tags' fields to dataset-hub-cache",
        )
        migration.up()

        assert_tags("dataset", kind="dataset-hub-cache")
        assert_unchanged("dataset_with_error", kind="dataset-hub-cache")

        migration.down()
        assert_unchanged("dataset", kind="dataset-hub-cache")
        assert_unchanged("dataset_with_error", kind="dataset-hub-cache")

        db[CACHE_COLLECTION_RESPONSES].drop()
