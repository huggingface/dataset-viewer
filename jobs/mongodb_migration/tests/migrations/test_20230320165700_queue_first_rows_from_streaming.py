# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230320165700_queue_first_rows_from_streaming import (
    MigrationQueueUpdateFirstRows,
)


def test_queue_update_first_rows_type_and_unicity_id(mongo_host: str) -> None:
    with MongoResource(database="test_queue_update_first_rows_type_and_unicity_id", host=mongo_host, mongoengine_alias="queue"):
        db = get_db("queue")
        db["jobsBlue"].insert_many([{"type": "/first-rows", "unicity_id": 'Job[/first-rows][dataset][config][split]', "dataset": "dataset", "http_status": 200}])
        assert db["jobsBlue"].find_one(
            {"type": "/first-rows"}
        )  # Ensure there exists at least one record to update

        migration = MigrationQueueUpdateFirstRows(
            version="20230320165700",
            description=(
                "update 'type' and 'unicity_id' fields in job from /first-rows to first-rows-from-streaming"
            ),
        )
        migration.up()

        assert not db["jobsBlue"].find_one(
            {"type": "/first-rows"}
        )  # Ensure that are not records with old kind

        result = db["jobsBlue"].find_one({"type": "first-rows-from-streaming"})
        assert result
        assert result["unicity_id"] == "Job[first-rows-from-streaming][dataset][config][split]"
        db["jobsBlue"].drop()
