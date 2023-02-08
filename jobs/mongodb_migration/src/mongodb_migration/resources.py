# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from libcommon.resources import DatabaseResource

from mongodb_migration.constants import DATABASE_MIGRATIONS_MONGOENGINE_ALIAS


class MigrationsDatabaseConnectionFailure(Exception):
    pass


@dataclass
class MigrationsDatabaseResource(DatabaseResource):
    """
    A resource that represents a connection to the migrations database.

    Args:
        database (:obj:`str`): The name of the database.
        host (:obj:`str`): The host of the database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=DATABASE_MIGRATIONS_MONGOENGINE_ALIAS, init=False)
