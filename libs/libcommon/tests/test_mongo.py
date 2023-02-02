# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.config import CacheConfig
from libcommon.mongo import CheckError, CreationError, MongoConnection


def test_mongo_connection(cache_config: CacheConfig) -> None:
    alias = "test"
    bad_host = "mongodb://doesnotexist:123"
    connection = MongoConnection(database="test", alias=alias, host=bad_host, serverSelectionTimeoutMS=200)

    # the host does not exist
    with pytest.raises(CheckError):
        connection.check_connection()

    # we cannot create two connections with the same alias
    with pytest.raises(CreationError):
        MongoConnection(database="test2", alias=alias, host=bad_host)

    # we can reuse the alias after the connection has been disconnected
    connection.disconnect()
    connection = MongoConnection(database="test2", alias=alias, host=bad_host)
    connection.disconnect()

    # we can connect to a real database (a test server is running in the CI)
    connection = MongoConnection(database=cache_config.mongo_database, alias=alias, host=cache_config.mongo_url)
    connection.check_connection()
    connection.disconnect()
