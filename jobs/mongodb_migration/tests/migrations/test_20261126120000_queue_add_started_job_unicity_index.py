# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import pytest
from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS, QUEUE_STARTED_JOB_UNICITY_INDEX_NAME
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pymongo.errors import DuplicateKeyError

from mongodb_migration.migrations._20261126120000_queue_add_started_job_unicity_index import (
    MigrationQueueAddStartedJobUnicityIndex,
)


def test_queue_add_started_job_unicity_index(mongo_host: str) -> None:
    with MongoResource(
        database="test_queue_add_started_job_unicity_index", host=mongo_host, mongoengine_alias="queue"
    ):
        collection = get_db(QUEUE_MONGOENGINE_ALIAS)[QUEUE_COLLECTION_JOBS]
        collection.insert_many(
            [
                {"unicity_id": "a", "status": "started"},
                {"unicity_id": "a", "status": "waiting"},
                {"unicity_id": "a", "status": "waiting"},
            ]
        )

        migration = MigrationQueueAddStartedJobUnicityIndex(
            version="20261126120000", description="add the unique index on the started jobs"
        )
        migration.up()
        migration.validate()
        index = collection.index_information()[QUEUE_STARTED_JOB_UNICITY_INDEX_NAME]
        assert index["key"] == [("unicity_id", 1)]
        assert index["unique"]
        assert index["partialFilterExpression"] == {"status": "started"}
        with pytest.raises(DuplicateKeyError):
            collection.update_one({"unicity_id": "a", "status": "waiting"}, {"$set": {"status": "started"}})

        # the index is created by hand before the deploy: the migration skips it
        migration.up()
        migration.validate()

        migration.down()
        assert QUEUE_STARTED_JOB_UNICITY_INDEX_NAME not in collection.index_information()
        with pytest.raises(ValueError):
            migration.validate()

        collection.drop()
