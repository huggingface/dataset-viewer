# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from libcommon.constants import QUEUE_COLLECTION_LOCKS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230622131500_lock_add_owner import (
    MigrationAddOwnerToQueueLock,
)


def assert_owner(key: str, owner: Optional[str]) -> None:
    db = get_db(QUEUE_MONGOENGINE_ALIAS)
    entry = db[QUEUE_COLLECTION_LOCKS].find_one({"key": key})
    assert entry is not None
    if owner is None:
        assert "owner" not in entry or entry["owner"] is None
    else:
        assert entry["owner"] == owner


def test_lock_add_owner(mongo_host: str) -> None:
    with MongoResource(database="test_lock_add_owner", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_LOCKS].insert_many(
            [
                {
                    "key": "key1",
                    "job_id": "job_id1",
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
                {
                    "key": "key2",
                    "job_id": None,
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
                {
                    "key": "key3",
                    "job_id": "job_id3",
                    "owner": "owner3",
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
            ]
        )

        migration = MigrationAddOwnerToQueueLock(
            version="20230622131500",
            description="add owner field to locks",
        )
        migration.up()

        assert_owner("key1", "job_id1")
        assert_owner("key2", None)
        assert_owner("key3", "owner3")

        migration.down()
        assert_owner("key1", None)
        assert_owner("key2", None)
        assert_owner("key3", None)

        db[QUEUE_COLLECTION_LOCKS].drop()
