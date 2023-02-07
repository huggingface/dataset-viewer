# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from mongoengine import Document
from mongoengine.fields import StringField

from mongodb_migration.resource import MigrationsDatabaseResource


def test_cache_database(mongo_host: str) -> None:
    resource = MigrationsDatabaseResource(database="test_migrations_database", host=mongo_host)
    resource.allocate()

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongo_connection.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    resource.release()
