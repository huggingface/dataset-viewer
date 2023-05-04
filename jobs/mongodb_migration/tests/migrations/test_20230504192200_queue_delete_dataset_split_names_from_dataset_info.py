# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.delete_migration import DeleteQueueMigration

QUEUE_MONGOENGINE_ALIAS = DeleteQueueMigration._MONGOENGINE_ALIAS
QUEUE_COLLECTION_JOBS = DeleteQueueMigration._COLLECTION_JOBS


def test_queue_delete_dataset_split_names_from_dataset_info(mongo_host: str) -> None:
    job_type = "dataset-split-names-from-dataset-info"
    with MongoResource(
        database="test_queue_dataset_split_names_from_dataset_info", host=mongo_host, mongoengine_alias="queue"
    ):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": job_type,
                    "unicity_id": f"Job[{job_type}][dataset][config][split]",
                    "dataset": "dataset",
                    "http_status": 200,
                }
            ]
        )
        assert db[QUEUE_COLLECTION_JOBS].find_one({"type": job_type})  # Ensure there is at least one record to delete

        migration = DeleteQueueMigration(
            job_type=job_type,
            version="20230504192800",
            description=f"remove jobs of type '{job_type}'",
        )
        migration.up()

        assert not db[QUEUE_COLLECTION_JOBS].find_one({"type": job_type})  # Ensure 0 records with old type

        db[QUEUE_COLLECTION_JOBS].drop()
