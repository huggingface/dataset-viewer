# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest
from mongoengine import Document
from mongoengine.fields import StringField
from pymongo.errors import ServerSelectionTimeoutError

from libcommon.mongo import MongoConnection, MongoConnectionFailure


def test_mongo_connection(queue_mongo_host: str) -> None:
    database_1 = "datasets_server_1"
    database_2 = "datasets_server_2"
    host = queue_mongo_host
    mongoengine_alias = "datasets_server_mongo_alias"
    server_selection_timeout_ms = 1_000
    connection_1 = MongoConnection(
        database=database_1,
        host=host,
        mongoengine_alias=mongoengine_alias,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )
    connection_2 = MongoConnection(
        database=database_2,
        host=host,
        mongoengine_alias=mongoengine_alias,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )

    connection_1.connect()
    with pytest.raises(MongoConnectionFailure):
        connection_2.connect()
    connection_1.disconnect()
    connection_2.connect()


@pytest.mark.parametrize(
    "host,mongoengine_alias,server_selection_timeout_ms,raises",
    [
        (None, "test_timeout_error", 5_000, False),
        ("mongodb://doesnotexist:123", "test_host_error", 5_000, True),
    ],
)
def test_mongo_connection_errors(
    queue_mongo_host: str,
    host: Optional[str],
    mongoengine_alias: str,
    server_selection_timeout_ms: int,
    raises: bool,
) -> None:
    if not host:
        host = queue_mongo_host
    database = "datasets_server_test"
    connection = MongoConnection(
        database=database,
        host=host,
        mongoengine_alias=mongoengine_alias,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )
    # this does not raise any issue, as it "only" registers the connection
    connection.connect()

    class User(Document):
        name = StringField()
        meta = {"db_alias": mongoengine_alias}

    if raises:
        with pytest.raises(ServerSelectionTimeoutError):
            len(User.objects())  # type: ignore
    else:
        assert len(User.objects()) == 0  # type: ignore
        # clean
        User.drop_collection()  # type: ignore

    connection.disconnect()
