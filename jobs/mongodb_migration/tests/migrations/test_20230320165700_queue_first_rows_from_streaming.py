# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230320165700_queue_first_rows_from_streaming import (
    MigrationQueueUpdateFirstRows,
)


def test_queue_update_first_rows_type_and_unicity_id(mongo_host: str) -> None:
    with MongoResource(
        database="test_queue_update_first_rows_type_and_unicity_id", host=mongo_host, mongoengine_alias="queue"
    ):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].insert_many(
            [
                {
                    "type": "/first-rows",
                    "unicity_id": "Job[/first-rows][dataset][config][split]",
                    "dataset": "dataset",
                    "http_status": 200,
                }
            ]
        )
        assert db[QUEUE_COLLECTION_JOBS].find_one(
            {"type": "/first-rows"}
        )  # Ensure there is at least one record to update

        migration = MigrationQueueUpdateFirstRows(
            version="20230320165700",
            description=(
                "update 'type' and 'unicity_id' fields in job from /first-rows to split-first-rows-from-streaming"
            ),
        )
        migration.up()

        assert not db[QUEUE_COLLECTION_JOBS].find_one({"type": "/first-rows"})  # Ensure 0 records with old type

        result = db[QUEUE_COLLECTION_JOBS].find_one({"type": "split-first-rows-from-streaming"})
        assert result
        assert result["unicity_id"] == "Job[split-first-rows-from-streaming][dataset][config][split]"
        db[QUEUE_COLLECTION_JOBS].drop()
