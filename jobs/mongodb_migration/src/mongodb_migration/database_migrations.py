# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from typing import Generic, Type, TypeVar

from mongoengine import Document
from mongoengine.fields import StringField
from mongoengine.queryset.queryset import QuerySet

from mongodb_migration.constants import (
    DATABASE_MIGRATIONS_COLLECTION_MIGRATIONS,
    DATABASE_MIGRATIONS_MONGOENGINE_ALIAS,
)

# START monkey patching ### hack ###
# see https://github.com/sbdchd/mongo-types#install
U = TypeVar("U", bound=Document)


def no_op(self, x):  # type: ignore
    return self


QuerySet.__class_getitem__ = types.MethodType(no_op, QuerySet)


class QuerySetManager(Generic[U]):
    def __get__(self, instance: object, cls: Type[U]) -> QuerySet[U]:
        return QuerySet(cls, cls._get_collection())


# END monkey patching ### hack ###


class DatabaseMigration(Document):
    """A database migration that has already been executed.

    Args:
        version (`str`): The version of the migration, with the format YYYYMMDDHHMMSS
        description (`str`): A description of the migration
    """

    meta = {
        "collection": DATABASE_MIGRATIONS_COLLECTION_MIGRATIONS,
        "db_alias": DATABASE_MIGRATIONS_MONGOENGINE_ALIAS,
    }
    version = StringField(required=True)
    description = StringField(required=True)

    objects = QuerySetManager["DatabaseMigration"]()


# only for the tests
def _clean_maintenance_database() -> None:
    """Delete all the jobs in the database"""
    DatabaseMigration.drop_collection()  # type: ignore
