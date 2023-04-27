# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230320163700_cache_first_rows_from_streaming import (
    MigrationCacheUpdateFirstRows,
)


def test_cache_update_first_rows_kind(mongo_host: str) -> None:
    with MongoResource(database="test_cache_update_first_rows_kind", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many([{"kind": "/first-rows", "dataset": "dataset", "http_status": 200}])
        assert db[CACHE_COLLECTION_RESPONSES].find_one(
            {"kind": "/first-rows"}
        )  # Ensure there is at least one record to update

        migration = MigrationCacheUpdateFirstRows(
            version="20230320163700",
            description="update 'kind' field in cache from /first-rows to split-first-rows-from-streaming",
        )
        migration.up()

        assert not db[CACHE_COLLECTION_RESPONSES].find_one({"kind": "/first-rows"})  # Ensure 0 records with old kind

        assert db[CACHE_COLLECTION_RESPONSES].find_one({"kind": "split-first-rows-from-streaming"})

        db[CACHE_COLLECTION_RESPONSES].drop()
