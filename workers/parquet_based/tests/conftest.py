# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Iterator

from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import _clean_cache_database
from pytest import MonkeyPatch, fixture

from parquet_based.config import AppConfig


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture
def set_env_vars() -> Iterator[MonkeyPatch]:
    mp = MonkeyPatch()
    mp.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    mp.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    yield mp
    mp.undo()


@fixture
def app_config(set_env_vars: MonkeyPatch) -> Iterator[AppConfig]:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    yield app_config
    # Clean the database after each test. Must be done in test databases only, ensured by the check above!
    # TODO: use a parameter to pass a reference to the database, instead of relying on the implicit global variable
    # managed by mongoengine
    _clean_cache_database()
    _clean_queue_database()
