# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.index_deletion_migrations import QueueIndexDeletionMigration
from mongodb_migration.migration import IrreversibleMigrationError

INDEX_DEFINITION = [("type", 1), ("dataset", 1), ("config", 1), ("split", 1), ("status", 1), ("priority", 1)]


def test_queue_index_deletion(mongo_host: str) -> None:
    with MongoResource(database="test_queue_index_deletion", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].create_index(INDEX_DEFINITION)

        migration = QueueIndexDeletionMigration(
            version="20230516101600",
            description="remove index without revision",
            index_definition=INDEX_DEFINITION,
        )

        assert (
            len(migration.get_index_names(db[QUEUE_COLLECTION_JOBS].index_information())) == 1
        )  # Ensure the indexes exists

        migration.up()

        assert (
            len(migration.get_index_names(db[QUEUE_COLLECTION_JOBS].index_information())) == 0
        )  # Ensure the indexes do not exist anymore

        with raises(IrreversibleMigrationError):
            migration.down()
        db[QUEUE_COLLECTION_JOBS].drop()
