# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path

import pytest
from mongoengine import Document
from mongoengine.fields import StringField

from libcommon.resource import (
    AssetsDirectoryResource,
    CacheDatabaseResource,
    LogResource,
    QueueDatabaseResource,
)


def test_log() -> None:
    resource = LogResource(log_level=logging.DEBUG)
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    resource.allocate()
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    # ^ This is a bug, the log level should be set to 10


def test_log_context_manager() -> None:
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    with LogResource(logging.DEBUG):
        assert logging.getLogger().getEffectiveLevel() == logging.WARNING
        # ^ This is a bug, the log level should be set to 10


@pytest.mark.parametrize("pass_directory", [True, False])
def test_assets_directory(tmp_path: Path, pass_directory: bool) -> None:
    storage_directory = str(tmp_path / "test_assets") if pass_directory else None
    resource = AssetsDirectoryResource(storage_directory=storage_directory)
    if pass_directory:
        assert resource.storage_directory == storage_directory
    assert resource.storage_directory is not None
    assert Path(resource.storage_directory).exists()


@pytest.mark.parametrize("pass_directory", [True, False])
def test_assets_directory_context_manager(tmp_path: Path, pass_directory: bool) -> None:
    storage_directory = str(tmp_path / "test_assets") if pass_directory else None
    with AssetsDirectoryResource(storage_directory=storage_directory) as resource:
        if pass_directory:
            assert resource.storage_directory == storage_directory
        assert resource.storage_directory is not None
        assert Path(resource.storage_directory).exists()


def test_cache_database(cache_mongo_host: str) -> None:
    resource = CacheDatabaseResource(database="test_cache_database", host=cache_mongo_host)
    resource.allocate()

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongo_connection.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    resource.release()


def test_queue_database(queue_mongo_host: str) -> None:
    resource = QueueDatabaseResource(database="test_queue_database", host=queue_mongo_host)
    resource.allocate()

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongo_connection.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    resource.release()
