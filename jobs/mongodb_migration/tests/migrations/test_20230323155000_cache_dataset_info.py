# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230323155000_cache_dataset_info import (
    MigrationCacheUpdateDatasetInfo,
)


def test_cache_update_dataset_info_kind(mongo_host: str) -> None:
    old_kind, new_kind = "/dataset-info", "dataset-info"
    with MongoResource(database="test_cache_update_dataset_info_kind", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many([{"kind": old_kind, "dataset": "dataset", "http_status": 200}])
        assert db[CACHE_COLLECTION_RESPONSES].find_one(
            {"kind": old_kind}
        )  # Ensure there is at least one record to update

        migration = MigrationCacheUpdateDatasetInfo(
            version="20230323155000",
            description=f"update 'kind' field in cache from {old_kind} to {new_kind}",
        )
        migration.up()

        assert not db[CACHE_COLLECTION_RESPONSES].find_one({"kind": old_kind})  # Ensure 0 records with old kind

        assert db[CACHE_COLLECTION_RESPONSES].find_one({"kind": new_kind})

        db[CACHE_COLLECTION_RESPONSES].drop()
