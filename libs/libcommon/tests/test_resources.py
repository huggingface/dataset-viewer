# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest
from mongoengine import Document
from mongoengine.fields import StringField
from pymongo.errors import ServerSelectionTimeoutError

from libcommon.resources import (
    CacheMongoResource,
    MongoConnectionFailure,
    MongoResource,
    QueueMongoResource,
)


def test_database_resource(queue_mongo_host: str) -> None:
    database_1 = "datasets_server_1"
    database_2 = "datasets_server_2"
    host = queue_mongo_host
    mongoengine_alias = "datasets_server_mongo_alias"
    server_selection_timeout_ms = 1_000
    resource_1 = MongoResource(
        database=database_1,
        host=host,
        mongoengine_alias=mongoengine_alias,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )
    assert resource_1.check()
    with pytest.raises(MongoConnectionFailure):
        MongoResource(
            database=database_2,
            host=host,
            mongoengine_alias=mongoengine_alias,
            server_selection_timeout_ms=server_selection_timeout_ms,
        )
    resource_1.release()
    resource_2 = MongoResource(
        database=database_2,
        host=host,
        mongoengine_alias=mongoengine_alias,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )
    resource_2.check()
    resource_2.release()


@pytest.mark.parametrize(
    "host,mongoengine_alias,server_selection_timeout_ms,raises",
    [
        (None, "test_timeout_error", 5_000, False),
        ("mongodb://doesnotexist:123", "test_host_error", 5_000, True),
    ],
)
def test_database_resource_errors(
    queue_mongo_host: str,
    host: Optional[str],
    mongoengine_alias: str,
    server_selection_timeout_ms: int,
    raises: bool,
) -> None:
    if not host:
        host = queue_mongo_host
    database = "datasets_server_test"
    resource = MongoResource(
        database=database,
        host=host,
        mongoengine_alias=mongoengine_alias,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )
    # ^ this does not raise any issue, as it "only" registers the connection

    class User(Document):
        name = StringField()
        meta = {"db_alias": mongoengine_alias}

    if raises:
        assert not resource.check()
        with pytest.raises(ServerSelectionTimeoutError):
            len(User.objects())  # type: ignore
    else:
        assert resource.check()
        assert len(User.objects()) == 0  # type: ignore
        # clean
        User.drop_collection()  # type: ignore

    resource.release()


def test_cache_database(cache_mongo_host: str) -> None:
    resource = CacheMongoResource(database="test_cache_database", host=cache_mongo_host)

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    assert resource.check()
    resource.release()


def test_queue_database(queue_mongo_host: str) -> None:
    resource = QueueMongoResource(database="test_queue_database", host=queue_mongo_host)

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    assert resource.check()
    resource.release()
