# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Iterator, List

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import _clean_queue_database
from libcommon.resources import (
    AssetsDirectoryResource,
    CacheDatabaseResource,
    QueueDatabaseResource,
)
from libcommon.simple_cache import _clean_cache_database
from pytest import MonkeyPatch, fixture

from admin.config import AppConfig

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.hub"]


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session(hf_endpoint: str, hf_token: str):
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", hf_endpoint)
    monkeypatch_session.setenv("COMMON_HF_TOKEN", hf_token)
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session")
def app_config(monkeypatch_session: MonkeyPatch) -> AppConfig:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


@fixture(scope="session")
def processing_steps(app_config: AppConfig) -> List[ProcessingStep]:
    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    return list(processing_graph.steps.values())


@fixture(scope="session")
def assets_directory(app_config: AppConfig) -> Iterator[str]:
    with AssetsDirectoryResource(storage_directory=app_config.assets.storage_directory) as resource:
        yield str(resource.storage_directory)


@fixture(autouse=True)
def cache_database_resource(app_config: AppConfig) -> Iterator[CacheDatabaseResource]:
    with CacheDatabaseResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture(autouse=True)
def queue_database_resource(app_config: AppConfig) -> Iterator[QueueDatabaseResource]:
    with QueueDatabaseResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url) as resource:
        yield resource
        _clean_queue_database()
