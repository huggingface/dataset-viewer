# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230516101500_queue_job_add_revision import (
    MigrationQueueAddRevisionToJob,
)


def test_queue_add_revision_to_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_queue_add_revision_to_jobs", host=mongo_host, mongoengine_alias="queue"):
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

        migration = MigrationQueueAddRevisionToJob(
            version="20230516101500",
            description="add revision field to jobs",
        )
        migration.up()

        result = list(db[QUEUE_COLLECTION_JOBS].find({"dataset": "test"}))
        assert len(result) == 1
        assert result[0]["revision"] == "main"

        migration.down()
        result = list(db[QUEUE_COLLECTION_JOBS].find({"dataset": "test"}))
        assert len(result) == 1
        assert "revision" not in result[0]

        db[QUEUE_COLLECTION_JOBS].drop()
