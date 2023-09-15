# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Optional, TypeVar

from mongoengine.connection import ConnectionFailure, connect, disconnect
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from libcommon.constants import (
    CACHE_MONGOENGINE_ALIAS,
    METRICS_MONGOENGINE_ALIAS,
    QUEUE_MONGOENGINE_ALIAS,
)

T = TypeVar("T", bound="Resource")


@dataclass
class Resource:
    """
    A resource that can be allocated and released.

    The method allocate() is called when the resource is created.
    The method release() allows to free the resource.

    It can be used as a context manager, in which case the resource is released when the context is exited.

    Example:
        >>> with Resource() as resource:
        ...     pass

    Resources should be inherited from this class and must implement the allocate(), check() and release() methods.
    """

    def __post_init__(self) -> None:
        self.allocate()

    def __enter__(self: T) -> T:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.release()

    def allocate(self) -> None:
        pass

    def release(self) -> None:
        pass


class MongoConnectionFailure(Exception):
    pass


@dataclass
class MongoResource(Resource):
    """
    A base resource that represents a connection to a database.

    The method is_available() allows to check if the resource is available. It's not called automatically.

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

    def allocate(self) -> None:
        try:
            self._client = connect(
                db=self.database,
                host=self.host,
                alias=self.mongoengine_alias,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
            )
        except ConnectionFailure as e:
            raise MongoConnectionFailure(f"Failed to connect to MongoDB: {e}") from e

    def is_available(self) -> bool:
        """Check if the connection is available."""
        try:
            self._client.is_mongos
            return True
        except ServerSelectionTimeoutError:
            return False

    def create_collection(self, document: Any) -> None:
        document.ensure_indexes()

    def enable_pre_and_post_images(self, collection_name: str) -> None:
        self._client[self.database].command(
            "collMod", collection_name, changeStreamPreAndPostImages={"enabled": True}
        )  # type: ignore

    def release(self) -> None:
        disconnect(alias=self.mongoengine_alias)

    def __reduce__(self) -> tuple[Any, ...]:
        # Needed to be able to use the resource in subprocesses in tests (e.g. tests/test_queue.py::test_lock).
        # This is because the _client in not picklable.
        return (MongoResource, (self.database, self.host, self.mongoengine_alias, self.server_selection_timeout_ms))


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


@dataclass
class MetricsMongoResource(MongoResource):
    """
    A resource that represents a connection to the metrics mongo database.

    Args:
        database (:obj:`str`): The name of the mongo database.
        host (:obj:`str`): The host of the mongo database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=METRICS_MONGOENGINE_ALIAS, init=False)
