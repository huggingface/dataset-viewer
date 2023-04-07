# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230407091400_queue_delete_splits import (
    MigrationQueueDeleteSplits,
)


def test_queue_remove_splits(mongo_host: str) -> None:
    job_type = "/splits"
    with MongoResource(database="test_queue_remove_splits", host=mongo_host, mongoengine_alias="queue"):
        db = get_db("queue")
        db["jobsBlue"].insert_many(
            [
                {
                    "type": job_type,
                    "unicity_id": f"Job[{job_type}][dataset][config][split]",
                    "dataset": "dataset",
                    "http_status": 200,
                }
            ]
        )
        assert db["jobsBlue"].find_one({"type": job_type})  # Ensure there is at least one record to delete

        migration = MigrationQueueDeleteSplits(
            version="20230407091400",
            description=f"remove jobs of type '{job_type}'",
        )
        migration.up()

        assert not db["jobsBlue"].find_one({"type": job_type})  # Ensure 0 records with old type

        db["jobsBlue"].drop()
