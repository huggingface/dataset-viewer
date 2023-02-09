# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path

import pytest
from mongoengine import Document
from mongoengine.fields import StringField

from libcommon.resources import (
    AssetsStorageAccessResource,
    CacheDatabaseResource,
    LogResource,
    QueueDatabaseResource,
)


def test_log() -> None:
    LogResource(log_level=logging.DEBUG)
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    # ^ This is a bug, the log level should be set to 10


def test_log_context_manager() -> None:
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    with LogResource(logging.DEBUG):
        assert logging.getLogger().getEffectiveLevel() == logging.WARNING
        # ^ This is a bug, the log level should be set to 10


@pytest.mark.parametrize("with_init", [True, False])
def test_assets_directory(tmp_path: Path, with_init: bool) -> None:
    directory = str(tmp_path / "test_assets") if with_init else None
    resource = AssetsStorageAccessResource(init_directory=directory)
    if with_init:
        assert resource.directory == directory
    assert resource.directory is not None
    assert Path(resource.directory).exists()


@pytest.mark.parametrize("with_init", [True, False])
def test_assets_directory_context_manager(tmp_path: Path, with_init: bool) -> None:
    directory = str(tmp_path / "test_assets") if with_init else None
    with AssetsStorageAccessResource(init_directory=directory) as resource:
        if with_init:
            assert resource.directory == directory
        assert resource.directory is not None
        assert Path(resource.directory).exists()


def test_cache_database(cache_mongo_host: str) -> None:
    resource = CacheDatabaseResource(database="test_cache_database", host=cache_mongo_host)

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongo_connection.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    resource.release()


def test_queue_database(queue_mongo_host: str) -> None:
    resource = QueueDatabaseResource(database="test_queue_database", host=queue_mongo_host)

    class User(Document):
        name = StringField()
        meta = {"db_alias": resource.mongo_connection.mongoengine_alias}

    assert len(User.objects()) == 0  # type: ignore
    # clean
    User.drop_collection()  # type: ignore
    resource.release()
