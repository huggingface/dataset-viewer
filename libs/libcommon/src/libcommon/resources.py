# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from mongoengine.connection import ConnectionFailure, connect, disconnect
from pymongo import MongoClient  # type: ignore
from pymongo.errors import ServerSelectionTimeoutError

from libcommon.constants import CACHE_MONGOENGINE_ALIAS, QUEUE_MONGOENGINE_ALIAS


@dataclass
class Resource:
    """
    A resource that can be allocated and released.

    The method allocate() is called when the resource is created. The method release() allows to free the resource.

    It can be used as a context manager, in which case the resource is released when the context is exited.

    Example:
        >>> with Resource() as resource:
        ...     pass

    Resources should be inherited from this class and implement the allocate() and release() methods.
    """

    def __post_init__(self):
        self.allocate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def allocate(self):
        pass

    def release(self):
        pass


class MongoConnectionFailure(Exception):
    pass


class MongoTimeoutError(Exception):
    pass


@dataclass
class MongoResource(Resource):
    """
    A base resource that represents a connection to a database.

    Args:
        database (:obj:`str`): The name of the mongo database.
        host (:obj:`str`): The host of the mongo database. It must start with ``mongodb://`` or ``mongodb+srv://``.
        mongoengine_alias (:obj:`str`): The alias of the connection in mongoengine.
        server_selection_timeout_ms (:obj:`int`, `optional`, defaults to 30_000): The timeout in milliseconds for
            server selection.
    """

    database: str
    host: str
    mongoengine_alias: str
    server_selection_timeout_ms: int = 30_000

    _client: MongoClient = field(init=False)

    def allocate(self):
        try:
            self._client = connect(
                db=self.database,
                host=self.host,
                alias=self.mongoengine_alias,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
            )
        except ConnectionFailure as e:
            raise MongoConnectionFailure(f"Failed to connect to MongoDB: {e}") from e

    def check(self) -> None:
        """Check if the connection works.
        Raises:
            - [`~libcommon.resources.MongoTimeoutError`]:
                if the mongo server could not be reached
        """
        try:
            self._client.is_mongos
        except ServerSelectionTimeoutError as e:
            raise MongoTimeoutError("Cannot connect to the mongo database server") from e

    def release(self):
        disconnect(alias=self.mongoengine_alias)


@dataclass
class CacheMongoResource(MongoResource):
    """
    A resource that represents a connection to the cache mongo database.

    Args:
        database (:obj:`str`): The name of the mongo database.
        host (:obj:`str`): The host of the mongo database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=CACHE_MONGOENGINE_ALIAS, init=False)


@dataclass
class QueueMongoResource(MongoResource):
    """
    A resource that represents a connection to the queue mongo database.

    Args:
        database (:obj:`str`): The name of the mongo database.
        host (:obj:`str`): The host of the mongo database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=QUEUE_MONGOENGINE_ALIAS, init=False)
