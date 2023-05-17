# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20230516101600_queue_delete_index_without_revision import (
    INDEX_DEFINITION,
    MigrationQueueDeleteIndexWithoutRevision,
    get_index_names,
)


def test_queue_delete_index_without_revision(mongo_host: str) -> None:
    with MongoResource(database="test_queue_delete_index_without_revision", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].create_index(INDEX_DEFINITION)
        assert len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information())) == 1  # Ensure the indexes exists

        migration = MigrationQueueDeleteIndexWithoutRevision(
            version="20230516101600",
            description="remove index without revision",
        )
        migration.up()

        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information())) == 0
        )  # Ensure the indexes do not exist anymore

        with raises(IrreversibleMigrationError):
            migration.down()
        db[QUEUE_COLLECTION_JOBS].drop()
