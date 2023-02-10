# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from libcommon.resources import MongoResource

from mongodb_migration.constants import DATABASE_MIGRATIONS_MONGOENGINE_ALIAS


@dataclass
class MigrationsMongoResource(MongoResource):
    """
    A resource that represents a connection to the migrations mongo database.

    Args:
        database (:obj:`str`): The name of the mongo database.
        host (:obj:`str`): The host of the mongo database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=DATABASE_MIGRATIONS_MONGOENGINE_ALIAS, init=False)
