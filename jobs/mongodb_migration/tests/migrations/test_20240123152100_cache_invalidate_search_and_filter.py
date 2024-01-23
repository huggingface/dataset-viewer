# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20240123152100_cache_invalidate_search_and_filter import (
    MigrationInvalidateSearchAndFilterCacheResponse,
)


def test_cache_invalidate_search_and_filter(mongo_host: str) -> None:
    with MongoResource(database="test_cache_invalidate_search_and_filter", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(
            [
                {
                    "kind": "dataset-is-valid",
                    "dataset": "dataset",
                    "http_status": 200,
                },
                {
                    "kind": "config-is-valid",
                    "dataset": "dataset",
                    "config": "config",
                    "http_status": 200,
                },
                {
                    "kind": "split-is-valid",
                    "dataset": "dataset",
                    "config": "config",
                    "split": "split",
                    "http_status": 200,
                },
                {
                    "kind": "split-is-valid",
                    "dataset": "dataset_error",
                    "config": "config_error",
                    "split": "split_error",
                    "http_status": 500,
                },
                {
                    "kind": "other-kind",
                    "dataset": "dataset",
                    "config": "config",
                    "content": {
                        "search": True,
                        "filter": True,
                    },
                    "split": "split",
                    "http_status": 200,
                },
            ]
        )

        migration = MigrationInvalidateSearchAndFilterCacheResponse(
            version="20240123152100",
            description="invalidate 'search' and 'filter' for dataset-is-valid, config-is-valid and split-is-valid cache entries",
        )

        migration.up()

        invalidated_cache = list(
            db[CACHE_COLLECTION_RESPONSES].find(
                {"kind": {"$in": ["dataset-is-valid", "config-is-valid", "split-is-valid"]}, "http_status": 200}
            )
        )
        assert len(invalidated_cache) == 3
        assert all(
            entry["content"]["search"] is False and entry["content"]["filter"] is False for entry in invalidated_cache
        )

        other_kind = list(db[CACHE_COLLECTION_RESPONSES].find({"kind": "other-kind"}))
        assert len(other_kind) == 1
        assert other_kind[0]["content"]["search"] is True and other_kind[0]["content"]["filter"] is True

        cache_with_error = list(
            db[CACHE_COLLECTION_RESPONSES].find(
                {"kind": {"$in": ["dataset-is-valid", "config-is-valid", "split-is-valid"]}, "http_status": 500}
            )
        )
        assert len(cache_with_error) == 1
        assert "content" not in cache_with_error[0]

        with raises(IrreversibleMigrationError):
            migration.down()

        db[CACHE_COLLECTION_RESPONSES].drop()
