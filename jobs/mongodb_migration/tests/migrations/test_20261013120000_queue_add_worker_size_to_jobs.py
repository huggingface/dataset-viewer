# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20261013120000_queue_add_worker_size_to_jobs import (
    INDEX_NAME,
    MigrationQueueAddWorkerSizeToJobs,
)


def test_queue_add_worker_size_to_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_queue_add_worker_size_to_jobs", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": "test",
                    "dataset": f"test/{difficulty}",
                    "revision": "123",
                    "unicity_id": f"test/{difficulty}",
                    "namespace": "test",
                    "created_at": "2022-01-01T00:00:00.000000Z",
                    "difficulty": difficulty,
                }
                for difficulty in [0, 20, 40, 41, 70, 71, 100]
            ]
        )

        migration = MigrationQueueAddWorkerSizeToJobs(
            version="20261013120000",
            description="add 'worker_size' field and index to the jobs",
        )
        migration.up()

        expected = {0: "light", 20: "light", 40: "light", 41: "medium", 70: "medium", 71: "heavy", 100: "heavy"}
        for result in db[QUEUE_COLLECTION_JOBS].find({}):
            assert result["worker_size"] == expected[result["difficulty"]]
        assert INDEX_NAME in db[QUEUE_COLLECTION_JOBS].index_information()

        migration.down()
        assert all("worker_size" not in result for result in db[QUEUE_COLLECTION_JOBS].find({}))
        assert INDEX_NAME not in db[QUEUE_COLLECTION_JOBS].index_information()

        db[QUEUE_COLLECTION_JOBS].drop()
