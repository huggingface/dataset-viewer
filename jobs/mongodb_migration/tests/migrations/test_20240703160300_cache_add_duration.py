# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240703160300_cache_add_duration import MigrationAddDurationToCacheResponse


def test_cache_add_retries_to_cache(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_duration_to_cache", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(
            [
                {
                    "kind": "kind",
                    "dataset": "test",
                    "http_status": 200,
                },
                {
                    "kind": "kind",
                    "dataset": "test2",
                    "http_status": 200,
                    "duration": 10.5,
                },
            ]
        )

        migration = MigrationAddDurationToCacheResponse(
            version="20240703160300", description="add 'duration' field to cache records"
        )
        migration.up()

        result = list(db[CACHE_COLLECTION_RESPONSES].find({"dataset": "test"}))
        assert len(result) == 1
        assert result[0]["duration"] is None

        result = list(db[CACHE_COLLECTION_RESPONSES].find({"dataset": "test2"}))
        assert len(result) == 1
        assert result[0]["duration"] == 10.5

        migration.down()
        result = list(db[CACHE_COLLECTION_RESPONSES].find())
        assert len(result) == 2
        assert "duration" not in result[0]
        assert "duration" not in result[1]

        db[CACHE_COLLECTION_RESPONSES].drop()
