# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240104085000_cache_add_retries import MigrationAddRetriesToCacheResponse


def test_cache_add_retries_to_cache(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_retries_to_cache", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_one(
            {
                "kind": "/splits",
                "dataset": "test",
                "http_status": 200,
                "worker_version": "1.0.0",
            }
        )

        migration = MigrationAddRetriesToCacheResponse(
            version="20240104085000", description="add 'retries' field to cache records"
        )
        migration.up()

        result = list(db[CACHE_COLLECTION_RESPONSES].find({"dataset": "test"}))
        assert len(result) == 1
        assert result[0]["retries"] == 0

        migration.down()
        result = list(db[CACHE_COLLECTION_RESPONSES].find({"dataset": "test"}))
        assert len(result) == 1
        assert "retries" not in result[0]

        db[CACHE_COLLECTION_RESPONSES].drop()
