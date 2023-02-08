# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import InitVar, dataclass, field
from typing import Optional

from libcommon.constants import CACHE_MONGOENGINE_ALIAS, QUEUE_MONGOENGINE_ALIAS
from libcommon.log import init_logging
from libcommon.mongo import MongoConnection, MongoConnectionFailure
from libcommon.storage import StrPath, init_dir


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


@dataclass
class AssetsDirectoryResource(Resource):
    """
    A resource that represents a directory where assets are stored.

    The directory is created if it does not exist.

    Args:
        init_storage_directory (:obj:`str`, optional): The path to the directory where assets are stored.
            If :obj:`None`, the directory is created in the ``datasets_server_assets`` subdirectory of the
            user cache directory.

    Example:
        >>> with AssetsDirectoryResource("/tmp/assets") as assets_directory_resource:
        ...     pass
    """

    init_storage_directory: InitVar[Optional[StrPath]] = None
    storage_directory: StrPath = field(init=False)

    def allocate(self):
        self.storage_directory = init_dir(directory=self.init_storage_directory, appname="datasets_server_assets")

    # no need to implement release() as the directory is not deleted


@dataclass
class LogResource(Resource):
    """
    A resource that sets the log level.

    Args:
        init_log_level (:obj:`int`, optional): The log level. If :obj:`None`, the log level is set to INFO.
    """

    init_log_level: InitVar[int] = None
    log_level: int = field(init=False)

    def allocate(self):
        self.log_level = init_logging(self.init_log_level)

    # no need to implement release() as the log level is not changed.
    # TODO: restore the previous log level?


class DatabaseConnectionFailure(Exception):
    pass


@dataclass
class DatabaseResource(Resource):
    """
    A base resource that represents a connection to a database.

    Args:
        database (:obj:`str`): The name of the database.
        host (:obj:`str`): The host of the database. It must start with ``mongodb://`` or ``mongodb+srv://``.
        mongoengine_alias (:obj:`str`): The alias of the connection in mongoengine.
    """

    database: str
    host: str
    mongoengine_alias: str

    mongo_connection: MongoConnection = field(init=False)

    def allocate(self):
        self.mongo_connection = MongoConnection(
            database=self.database, host=self.host, mongoengine_alias=self.mongoengine_alias
        )
        try:
            self.mongo_connection.connect()
        except MongoConnectionFailure as e:
            raise DatabaseConnectionFailure(f"Failed to connect to the database: {e}") from e

    def release(self):
        self.mongo_connection.disconnect()


@dataclass
class CacheDatabaseResource(DatabaseResource):
    """
    A resource that represents a connection to the cache database.

    Args:
        database (:obj:`str`): The name of the database.
        host (:obj:`str`): The host of the database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=CACHE_MONGOENGINE_ALIAS, init=False)


@dataclass
class QueueDatabaseResource(DatabaseResource):
    """
    A resource that represents a connection to the queue database.

    Args:
        database (:obj:`str`): The name of the database.
        host (:obj:`str`): The host of the database. It must start with ``mongodb://`` or ``mongodb+srv://``.
    """

    mongoengine_alias: str = field(default=QUEUE_MONGOENGINE_ALIAS, init=False)
