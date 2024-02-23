# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.drop_migrations import MigrationDropCollection
from mongodb_migration.migration import IrreversibleMigrationError

TEST_DATABASE = "test_database"
TEST_COLLECTION = "test_collection"


def test_drop_collection(mongo_host: str) -> None:
    with MongoResource(database="test_drop_collection", host=mongo_host, mongoengine_alias=TEST_DATABASE):
        db = get_db(TEST_DATABASE)
        db[TEST_COLLECTION].insert_many(
            [
                {
                    "kind": "kind",
                    "error_code": "UnexpectedError",
                    "http_status": 500,
                    "total": 1,
                }
            ]
        )
        assert db[TEST_COLLECTION].find_one({"kind": "kind"}) is not None
        assert TEST_COLLECTION in db.list_collection_names()  # type: ignore

        migration = MigrationDropCollection(
            version="20230811063600",
            description="drop collection",
            alias=TEST_DATABASE,
            collection_name=TEST_COLLECTION,
        )

        migration.up()
        assert db[TEST_COLLECTION].find_one({"kind": "kind"}) is None
        assert TEST_COLLECTION not in db.list_collection_names()  # type: ignore

        with raises(IrreversibleMigrationError):
            migration.down()
