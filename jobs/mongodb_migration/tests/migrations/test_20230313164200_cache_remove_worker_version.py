# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20230313164200_cache_remove_worker_version import (
    MigrationRemoveWorkerVersionFromCachedResponse,
)


def test_cache_remove_worker_version(mongo_host: str) -> None:
    with MongoResource(database="test_cache_remove_worker_version", host=mongo_host, mongoengine_alias="cache"):
        db = get_db("cache")
        db["cachedResponsesBlue"].delete_many({})
        db["cachedResponsesBlue"].insert_many(
            [
                {
                    "kind": "/splits",
                    "dataset": "dataset_without_worker_version",
                    "http_status": 200,
                    "worker_version": "1.0.0",
                }
            ]
        )
        migration = MigrationRemoveWorkerVersionFromCachedResponse(
            version="20230313164200", description="remove 'worker_version' field from cache"
        )
        migration.up()
        result = db["cachedResponsesBlue"].find_one({"dataset": "dataset_without_worker_version"})
        assert result
        assert "worker_version" not in result

        with raises(IrreversibleMigrationError):
            migration.down()
        db["cachedResponsesBlue"].drop()
