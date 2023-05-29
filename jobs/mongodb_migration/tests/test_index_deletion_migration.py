# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_MONGOENGINE_ALIAS,
    QUEUE_COLLECTION_JOBS,
    QUEUE_MONGOENGINE_ALIAS,
)
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.index_deletion_migrations import (
    CacheIndexDeletionMigration,
    QueueIndexDeletionMigration,
)
from mongodb_migration.migration import IrreversibleMigrationError


def test_queue_index_deletion(mongo_host: str) -> None:
    with MongoResource(
        database="test_queue_index_deletion", host=mongo_host, mongoengine_alias=QUEUE_MONGOENGINE_ALIAS
    ):
        index_definition = [("type", 1), ("dataset", 1), ("config", 1), ("split", 1), ("status", 1), ("priority", 1)]

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].create_index(index_definition)

        migration = QueueIndexDeletionMigration(
            version="20230529103700",
            description="remove index",
            index_definition=index_definition,
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


def test_cache_index_deletion(mongo_host: str) -> None:
    with MongoResource(
        database="test_cache_index_deletion", host=mongo_host, mongoengine_alias=CACHE_MONGOENGINE_ALIAS
    ):
        index_definition = [("kind", 1), ("dataset", 1), ("config", 1), ("split", 1)]
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].create_index(index_definition)

        migration = CacheIndexDeletionMigration(
            version="20230529111600",
            description="remove index",
            index_definition=index_definition,
        )

        assert (
            len(migration.get_index_names(db[CACHE_COLLECTION_RESPONSES].index_information())) == 1
        )  # Ensure the indexes exists

        migration.up()

        assert (
            len(migration.get_index_names(db[CACHE_COLLECTION_RESPONSES].index_information())) == 0
        )  # Ensure the indexes do not exist anymore

        with raises(IrreversibleMigrationError):
            migration.down()
        db[CACHE_COLLECTION_RESPONSES].drop()
