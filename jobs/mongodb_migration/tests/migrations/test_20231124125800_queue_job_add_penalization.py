# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20231124125800_queue_job_add_penalization import MigrationQueueAddPenalizationToJob


def test_queue_add_penalization_to_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_queue_add_penalization_to_jobs", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": "test",
                    "dataset": "test",
                    "revision": "123",
                    "unicity_id": "test",
                    "namespace": "test",
                    "created_at": "2022-01-01T00:00:00.000000Z",
                }
            ]
        )

        migration = MigrationQueueAddPenalizationToJob(
            version="20231124125800",
            description="add penalization field to jobs",
        )
        migration.up()

        result = db[QUEUE_COLLECTION_JOBS].find_one({"dataset": "test"})
        assert result
        assert result["penalization"] == 0

        migration.down()
        result = db[QUEUE_COLLECTION_JOBS].find_one({"dataset": "test"})
        assert result
        assert "penalization" not in result

        db[QUEUE_COLLECTION_JOBS].drop()
