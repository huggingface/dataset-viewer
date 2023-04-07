# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230407091500_cache_delete_splits import (
    MigrationCacheDeleteSplits,
)


def test_cache_remove_splits(mongo_host: str) -> None:
    kind = "/splits"
    with MongoResource(database="test_cache_remove_splits", host=mongo_host, mongoengine_alias="cache"):
        db = get_db("cache")
        db["cachedResponsesBlue"].insert_many([{"kind": kind, "dataset": "dataset", "http_status": 200}])
        assert db["cachedResponsesBlue"].find_one({"kind": kind})  # Ensure there is at least one record to update

        migration = MigrationCacheDeleteSplits(
            version="20230407091500",
            description=f"remove cache for kind {kind}",
        )
        migration.up()

        assert not db["cachedResponsesBlue"].find_one({"kind": kind})  # Ensure 0 records with old kind

        db["cachedResponsesBlue"].drop()
