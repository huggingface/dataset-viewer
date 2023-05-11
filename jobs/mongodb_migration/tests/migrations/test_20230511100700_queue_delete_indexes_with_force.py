# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.queue import Job
from libcommon.resources import MongoResource
from libcommon.utils import get_datetime
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230511100700_queue_delete_indexes_with_force import (
    MigrationQueueDeleteIndexesWithForce,
    field_name,
    get_index_names,
)


def test_queue_delete_indexes_with_force(mongo_host: str) -> None:
    with MongoResource(database="test_queue_delete_indexes_with_force", host=mongo_host, mongoengine_alias="queue"):
        Job(type="test", dataset="test", unicity_id="test", namespace="test", created_at=get_datetime()).save()
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].create_index(field_name)
        db[QUEUE_COLLECTION_JOBS].create_index([(field_name, 1), ("type", 1)])
        db[QUEUE_COLLECTION_JOBS].create_index([("type", 1), (field_name, 1)])
        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information(), "force")) == 3
        )  # Ensure the TTL index exists

        migration = MigrationQueueDeleteIndexesWithForce(
            version="20230511100700",
            description="remove indexes with field 'force'",
        )
        migration.up()

        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information(), "force")) == 0
        )  # Ensure the TTL index exists  # Ensure 0 records with old type

        db[QUEUE_COLLECTION_JOBS].drop()
