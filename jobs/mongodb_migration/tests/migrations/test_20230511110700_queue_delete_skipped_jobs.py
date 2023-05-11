# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20230511110700_queue_delete_skipped_jobs import (
    MigrationDeleteSkippedJobs,
    status,
)


def test_queue_delete_skipped_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_delete_skipped_jobs", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].delete_many({})
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": "test",
                    "dataset": "dataset",
                    "status": status,
                },
                {
                    "type": "test",
                    "dataset": "dataset",
                    "status": "waiting",
                },
                {
                    "type": "test",
                    "dataset": "dataset",
                    "status": status,
                },
                {
                    "type": "test",
                    "dataset": "dataset",
                    "status": "started",
                },
            ]
        )
        migration = MigrationDeleteSkippedJobs(
            version="20230511110700", description=f"delete jobs with {status} status"
        )
        migration.up()
        result = list(db[QUEUE_COLLECTION_JOBS].find({"dataset": "dataset"}))
        assert len(result) == 2
        assert all(doc["status"] != status for doc in result)

        with raises(IrreversibleMigrationError):
            migration.down()
        db[QUEUE_COLLECTION_JOBS].drop()
