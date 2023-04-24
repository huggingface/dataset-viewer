# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230424173000_cache_delete_parquet_and_dataset_info import (
    MigrationCacheDeleteParquetAndDatasetInfo,
)


def test_cache_delete_parquet_and_dataset_info(mongo_host: str) -> None:
    kind = "/parquet-and-dataset-info"
    with MongoResource(
        database="test_cache_delete_parquet_and_dataset_info", host=mongo_host, mongoengine_alias="cache"
    ):
        db = get_db("cache")
        db["cachedResponsesBlue"].insert_many([{"kind": kind, "dataset": "dataset", "http_status": 200}])
        assert db["cachedResponsesBlue"].find_one({"kind": kind})  # Ensure there is at least one record to update

        migration = MigrationCacheDeleteParquetAndDatasetInfo(
            version="20230424173000",
            description=f"remove cache for kind {kind}",
        )
        migration.up()

        assert not db["cachedResponsesBlue"].find_one({"kind": kind})  # Ensure 0 records with old kind

        db["cachedResponsesBlue"].drop()
