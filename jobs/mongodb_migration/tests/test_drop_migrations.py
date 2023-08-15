# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import CACHE_METRICS_COLLECTION, METRICS_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db
from pytest import raises

from mongodb_migration.drop_migrations import MigrationDropCollection
from mongodb_migration.migration import IrreversibleMigrationError


def test_drop_collection(mongo_host: str) -> None:
    with MongoResource(database="test_drop_collection", host=mongo_host, mongoengine_alias=METRICS_MONGOENGINE_ALIAS):
        db = get_db(METRICS_MONGOENGINE_ALIAS)
        db[CACHE_METRICS_COLLECTION].insert_many(
            [
                {
                    "kind": "kind",
                    "error_code": "UnexpectedError",
                    "http_status": 500,
                    "total": 1,
                }
            ]
        )
        assert db[CACHE_METRICS_COLLECTION].find_one({"kind": "kind"}) is not None
        assert CACHE_METRICS_COLLECTION in db.list_collection_names()  # type: ignore

        migration = MigrationDropCollection(
            version="20230811063600",
            description="drop cache metrics collection",
            alias=METRICS_MONGOENGINE_ALIAS,
            collection_name=CACHE_METRICS_COLLECTION,
        )

        migration.up()
        assert db[CACHE_METRICS_COLLECTION].find_one({"kind": "kind"}) is None
        assert CACHE_METRICS_COLLECTION not in db.list_collection_names()  # type: ignore

        with raises(IrreversibleMigrationError):
            migration.down()
