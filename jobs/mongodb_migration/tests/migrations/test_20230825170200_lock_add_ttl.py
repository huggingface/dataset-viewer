# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from libcommon.constants import QUEUE_COLLECTION_LOCKS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230825170200_lock_add_ttl import (
    MigrationAddTtlToQueueLock,
)


def assert_ttl(key: str, ttl: Optional[int]) -> None:
    db = get_db(QUEUE_MONGOENGINE_ALIAS)
    entry = db[QUEUE_COLLECTION_LOCKS].find_one({"key": key})
    assert entry is not None
    if ttl is None:
        assert "ttl" not in entry or entry["ttl"] is None
    else:
        assert entry["ttl"] == ttl


def test_lock_add_ttl(mongo_host: str) -> None:
    with MongoResource(database="test_lock_add_ttl", host=mongo_host, mongoengine_alias="queue"):
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_LOCKS].insert_many(
            [
                {
                    "key": "key1",
                    "owner": "job_id1",
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
                {
                    "key": "key2",
                    "owner": None,
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
                {
                    "key": "key3",
                    "owner": "job_id3",
                    "ttl": 600,
                    "created_at": "2022-01-01T00:00:00.000000Z",
                },
            ]
        )

        migration = MigrationAddTtlToQueueLock(
            version="20230825170200",
            description="add ttl field to locks",
        )
        migration.up()

        assert_ttl("key1", None)
        assert_ttl("key2", None)
        assert_ttl("key3", 600)

        migration.down()
        assert_ttl("key1", None)
        assert_ttl("key2", None)
        assert_ttl("key3", None)

        db[QUEUE_COLLECTION_LOCKS].drop()
