# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230323160000_queue_dataset_info import (
    MigrationQueueUpdateDatasetInfo,
)


def test_queue_update_dataset_info_type_and_unicity_id(mongo_host: str) -> None:
    old_kind, new_kind = "/dataset-info", "dataset-info"
    with MongoResource(
        database="test_queue_update_dataset_info_type_and_unicity_id", host=mongo_host, mongoengine_alias="queue"
    ):
        db = get_db("queue")
        db["jobsBlue"].insert_many(
            [
                {
                    "type": old_kind,
                    "unicity_id": f"Job[{old_kind}][dataset][config][split]",
                    "dataset": "dataset",
                    "http_status": 200,
                }
            ]
        )
        assert db["jobsBlue"].find_one({"type": old_kind})  # Ensure there is at least one record to update

        migration = MigrationQueueUpdateDatasetInfo(
            version="20230323160000",
            description=f"update 'type' and 'unicity_id' fields in job from {old_kind} to {new_kind}",
        )
        migration.up()

        assert not db["jobsBlue"].find_one({"type": old_kind})  # Ensure 0 records with old type

        result = db["jobsBlue"].find_one({"type": new_kind})
        assert result
        assert result["unicity_id"] == f"Job[{new_kind}][dataset][config][split]"
        db["jobsBlue"].drop()
