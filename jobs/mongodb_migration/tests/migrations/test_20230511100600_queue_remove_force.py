# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20230511100600_queue_remove_force import (
    MigrationRemoveForceFromJob,
)


def test_cache_remove_worker_version(mongo_host: str) -> None:
    with MongoResource(database="test_queue_remove_force", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].delete_many({})
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": "test",
                    "dataset": "dataset_without_force",
                    "force": True,
                }
            ]
        )
        migration = MigrationRemoveForceFromJob(
            version="20230511100600", description="remove 'force' field from queue"
        )
        migration.up()
        result = db[QUEUE_COLLECTION_JOBS].find_one({"dataset": "dataset_without_force"})
        assert result
        assert "force" not in result

        with raises(IrreversibleMigrationError):
            migration.down()
        db[QUEUE_COLLECTION_JOBS].drop()
