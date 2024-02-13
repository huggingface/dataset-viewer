# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240112164500_cache_add_partial_field_in_split_descriptive_statistics import (
    MigrationAddPartialToSplitDescriptiveStatisticsCacheResponse,
)


def assert_has_bool_partial_field(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "partial" in entry["content"]
    assert isinstance(entry["content"]["partial"], bool)


def assert_unchanged(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "partial" not in entry["content"]


def test_cache_add_partial(mongo_host: str) -> None:
    kind = "split-descriptive-statistics"
    with MongoResource(database="test_cache_add_partial", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        cache: list[dict[str, Any]] = [
            {
                "config": "default",
                "dataset": "dataset",
                "kind": kind,
                "split": "train",
                "content": {
                    "num_examples": 20,
                    "statistics": {},
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": 1,
            },
            {
                "config": "default",
                "dataset": "dataset_with_error",
                "kind": kind,
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
                "progress": 1,
            },
        ]

        db[CACHE_COLLECTION_RESPONSES].insert_many(cache)

        migration = MigrationAddPartialToSplitDescriptiveStatisticsCacheResponse(
            version="20240112164500",
            description="add 'partial' fields for 'split-descriptive-statistics' cache records. ",
        )
        migration.up()

        assert_has_bool_partial_field("dataset", kind=kind)
        assert_unchanged("dataset_with_error", kind=kind)

        migration.down()
        assert_unchanged("dataset", kind=kind)
        assert_unchanged("dataset_with_error", kind=kind)

        db[CACHE_COLLECTION_RESPONSES].drop()
