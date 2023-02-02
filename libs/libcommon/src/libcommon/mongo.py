# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from mongoengine.connection import ConnectionFailure, connect, disconnect
from pymongo.errors import ServerSelectionTimeoutError
from pymongo import MongoClient  # type: ignore


class CreationError(Exception):
    """Raised when the database connection can't be created."""

    pass


class CheckError(Exception):
    """Raised when the database connection check fails."""

    pass


@dataclass
class MongoConnection:
    """A class to connect to a MongoDB database.

    Args:
        database (`str`): The name of the database
        alias (`str`): The alias of the database
        host (`str`): The host of the database
        serverSelectionTimeoutMS (`int`, *optional*): The timeout for the connection in milliseconds. Defaults to
            30_000.

    Raises:
        - [`~libcommon.mongo.CreationError`]:
            if the database connection can't be created
    """

    database: str
    alias: str
    host: str
    serverSelectionTimeoutMS: Optional[int] = 30_000

    _client: MongoClient = field(init=False)

    def __post_init__(self) -> None:
        """Connect to the database.

        Raises:
            - [`~libcommon.mongo.CreationError`]:
                if the database connection can't be established
        """
        try:
            self._client = connect(
                db=self.database,
                alias=self.alias,
                host=self.host,
                serverSelectionTimeoutMS=self.serverSelectionTimeoutMS,
            )
        except ConnectionFailure as e:
            raise CreationError("Can't create the connection to the database") from e

    def check_connection(self) -> None:
        """Check if the connection works.

        Raises:
            - [`~libcommon.mongo.CheckError`]:
                if the database connection check fails
        """
        try:
            self._client.is_mongos
        except ServerSelectionTimeoutError as e:
            raise CheckError("Cannot connect to the database server") from e

    def disconnect(self) -> None:
        """Disconnect from the database."""
        disconnect(alias=self.alias)
