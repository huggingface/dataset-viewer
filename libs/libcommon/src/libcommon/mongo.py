# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass

from mongoengine.connection import ConnectionFailure, connect, disconnect


class MongoConnectionFailure(Exception):
    pass


@dataclass
class MongoConnection:
    """
    A connection to a MongoDB database.

    Args:
        database (:obj:`str`): The name of the database.
        host (:obj:`str`): The host of the database.
        mongoengine_alias (:obj:`str`): The alias of the connection for MongoEngine.
        server_selection_timeout_ms (:obj:`int`, `optional`, defaults to 30_000): The timeout in milliseconds for
            server selection.
    """

    database: str
    host: str
    mongoengine_alias: str
    server_selection_timeout_ms: int = 30_000

    def connect(self):
        try:
            connect(
                db=self.database,
                host=self.host,
                alias=self.mongoengine_alias,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
            )
        except ConnectionFailure as e:
            raise MongoConnectionFailure(f"Failed to connect to MongoDB: {e}") from e

    def disconnect(self):
        disconnect(alias=self.mongoengine_alias)
