# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230428181800_queue_delete_dataset_split_names_from_streaming import (
    MigrationQueueDeleteDatasetSplitNamesFromStreaming,
)


def test_queue_delete_dataset_split_names_from_streaming(mongo_host: str) -> None:
    job_type = "dataset-split-names-from-streaming"
    with MongoResource(
        database="test_queue_dataset_split_names_from_streaming", host=mongo_host, mongoengine_alias="queue"
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

        migration = MigrationQueueDeleteDatasetSplitNamesFromStreaming(
            version="20230428190400",
            description=f"remove jobs of type '{job_type}'",
        )
        migration.up()

        assert not db[QUEUE_COLLECTION_JOBS].find_one({"type": job_type})  # Ensure 0 records with old type

        db[QUEUE_COLLECTION_JOBS].drop()
