# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MIN_DIFFICULTY_INDEX_NAME, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20261125160000_cache_add_min_difficulty_index import (
    MigrationCacheAddMinDifficultyIndex,
)


def test_cache_add_min_difficulty_index(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_min_difficulty_index", host=mongo_host, mongoengine_alias="cache"):
        collection = get_db(CACHE_MONGOENGINE_ALIAS)[CACHE_COLLECTION_RESPONSES]
        collection.insert_one({"kind": "test", "dataset": "test", "http_status": 200})

        migration = MigrationCacheAddMinDifficultyIndex(
            version="20261125160000", description="add the 'min_difficulty' partial index to the cache"
        )
        migration.up()
        migration.validate()
        index = collection.index_information()[CACHE_MIN_DIFFICULTY_INDEX_NAME]
        assert index["key"] == [("dataset", 1), ("kind", 1)]
        assert index["partialFilterExpression"] == {"min_difficulty": {"$exists": True}}

        migration.down()
        assert CACHE_MIN_DIFFICULTY_INDEX_NAME not in collection.index_information()

        collection.drop()
