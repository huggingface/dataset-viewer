# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from mongoengine import Document
from mongoengine.fields import StringField

from mongodb_migration.resources import MigrationsMongoResource


def test_cache_database(mongo_host: str) -> None:
    resource = MigrationsMongoResource(database="test_migrations_database", host=mongo_host)

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    assert resource.is_available()
    resource.release()
