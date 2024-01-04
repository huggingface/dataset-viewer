# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240104084900_queue_job_add_attempts import (
    MigrationAddAttemptsToJob,
)


def test_queue_add_attempts_to_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_queue_add_attempts_to_jobs", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_one(
            {
                "type": "test",
                "dataset": "test",
                "unicity_id": "test",
                "namespace": "test",
                "created_at": "2022-01-01T00:00:00.000000Z",
            }
        )

        migration = MigrationAddAttemptsToJob(
            version="20240104084900", description="add 'attempts' field to jobs in queue database"
        )
        migration.up()

        result = list(db[QUEUE_COLLECTION_JOBS].find({"dataset": "test"}))
        assert len(result) == 1
        assert result[0]["attempts"] == 0

        migration.down()
        result = list(db[QUEUE_COLLECTION_JOBS].find({"dataset": "test"}))
        assert len(result) == 1
        assert "attempts" not in result[0]

        db[QUEUE_COLLECTION_JOBS].drop()
