# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230424173000_queue_delete_parquet_and_dataset_info import (
    MigrationQueueDeleteParquetAndDatasetInfo,
)


def test_queue_delete_parquet_and_dataset_info(mongo_host: str) -> None:
    job_type = "/parquet-and-dataset-info"
    with MongoResource(
        database="test_queue_delete_parquet_and_dataset_info", host=mongo_host, mongoengine_alias="queue"
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

        migration = MigrationQueueDeleteParquetAndDatasetInfo(
            version="20230424173000",
            description=f"remove jobs of type '{job_type}'",
        )
        migration.up()

        assert not db[QUEUE_COLLECTION_JOBS].find_one({"type": job_type})  # Ensure 0 records with old type

        db[QUEUE_COLLECTION_JOBS].drop()
