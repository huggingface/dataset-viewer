# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230705160600_queue_job_add_difficulty import (
    MigrationQueueAddDifficultyToJob,
)


def test_queue_add_difficulty_to_jobs(mongo_host: str) -> None:
    with MongoResource(database="test_queue_add_difficulty_to_jobs", host=mongo_host, mongoengine_alias="queue"):
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
                },
                {
                    "type": "dataset-is-valid",
                    "dataset": "test",
                    "revision": "123",
                    "unicity_id": "test",
                    "namespace": "test",
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
            ]
        )

        migration = MigrationQueueAddDifficultyToJob(
            version="20230705160600",
            description="add difficulty field to jobs",
        )
        migration.up()

        results = list(db[QUEUE_COLLECTION_JOBS].find({"type": "test"}))
        assert len(results) == 1
        assert results[0]["difficulty"] == 50
        results = list(db[QUEUE_COLLECTION_JOBS].find({"type": "dataset-is-valid"}))
        assert len(results) == 1
        assert results[0]["difficulty"] == 20

        migration.down()
        results = list(db[QUEUE_COLLECTION_JOBS].find({"dataset": "test"}))
        assert len(results) == 2
        assert all("difficulty" not in result for result in results)

        db[QUEUE_COLLECTION_JOBS].drop()
