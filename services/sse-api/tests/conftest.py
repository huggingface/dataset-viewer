# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
from typing import Iterator

import pytest
from environs import Env
from libcommon.constants import CACHE_COLLECTION_RESPONSES
from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import CachedResponseDocument, _clean_cache_database

from sse_api.config import AppConfig


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@pytest.fixture(scope="session")
def monkeypatch_session() -> Iterator[pytest.MonkeyPatch]:
    monkeypatch_session = pytest.MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    hostname = "localhost"
    port = "8888"
    monkeypatch_session.setenv("API_HF_TIMEOUT_SECONDS", "10")
    monkeypatch_session.setenv("API_UVICORN_HOSTNAME", hostname)
    monkeypatch_session.setenv("API_UVICORN_PORT", port)
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", f"http://{hostname}:{port}")
    yield monkeypatch_session
    monkeypatch_session.undo()


@pytest.fixture(scope="session")
def app_config(monkeypatch_session: pytest.MonkeyPatch) -> AppConfig:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


@pytest.fixture(scope="session")
def env() -> Env:
    return Env(expand_vars=True)


@pytest.fixture(scope="session")
def cache_mongo_host(env: Env) -> str:
    try:
        url = env.str(name="CACHE_MONGO_URL")
        if type(url) is not str:
            raise ValueError("CACHE_MONGO_URL is not set")
        return url
    except Exception as e:
        raise ValueError("CACHE_MONGO_URL is not set") from e


@pytest.fixture(scope="function")
def cache_mongo_resource(cache_mongo_host: str) -> Iterator[CacheMongoResource]:
    database = "datasets_server_cache_test"
    host = cache_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with CacheMongoResource(database=database, host=host) as cache_mongo_resource:
        _clean_cache_database()
        cache_mongo_resource.create_collection(CachedResponseDocument)
        cache_mongo_resource.enable_pre_and_post_images(CACHE_COLLECTION_RESPONSES)
        yield cache_mongo_resource
        _clean_cache_database()
        cache_mongo_resource.release()


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """
    Create an instance of the default event loop for each test case.

    See https://github.com/pytest-dev/pytest-asyncio/issues/38#issuecomment-264418154
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
