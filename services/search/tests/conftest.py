# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Iterator

from libapi.config import UvicornConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database
from libcommon.storage import (
    StrPath,
    init_cached_assets_dir,
    init_duckdb_index_cache_dir,
)
from pytest import MonkeyPatch, fixture

from search.config import AppConfig


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session() -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("CACHED_ASSETS_BASE_URL", "http://localhost/cached-assets")
    hostname = "localhost"
    port = "8888"
    monkeypatch_session.setenv("API_HF_TIMEOUT_SECONDS", "10")
    monkeypatch_session.setenv("API_UVICORN_HOSTNAME", hostname)
    monkeypatch_session.setenv("API_UVICORN_PORT", port)
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", f"http://{hostname}:{port}")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session")
def app_config(monkeypatch_session: MonkeyPatch) -> AppConfig:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


@fixture(scope="session")
def processing_graph(app_config: AppConfig) -> ProcessingGraph:
    return ProcessingGraph(app_config.processing_graph)


@fixture(scope="session")
def search_endpoint() -> str:
    return "/search"


@fixture(autouse=True)
def cache_mongo_resource(app_config: AppConfig) -> Iterator[CacheMongoResource]:
    with CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture(autouse=True)
def queue_mongo_resource(app_config: AppConfig) -> Iterator[QueueMongoResource]:
    with QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url) as resource:
        yield resource
        _clean_queue_database()


@fixture(scope="session")
def uvicorn_config(monkeypatch_session: MonkeyPatch) -> UvicornConfig:
    return UvicornConfig.from_env()


@fixture(scope="session")
def httpserver_listen_address(uvicorn_config: UvicornConfig) -> tuple[str, int]:
    return (uvicorn_config.hostname, uvicorn_config.port)


@fixture(scope="session")
def hf_endpoint(app_config: AppConfig) -> str:
    return app_config.common.hf_endpoint


@fixture(scope="session")
def hf_auth_path(app_config: AppConfig) -> str:
    return app_config.api.hf_auth_path


@fixture
def cached_assets_directory(app_config: AppConfig) -> StrPath:
    return init_cached_assets_dir(app_config.cached_assets.storage_directory)


@fixture
def duckdb_index_cache_directory(app_config: AppConfig) -> StrPath:
    return init_duckdb_index_cache_dir(app_config.duckdb_index.cache_directory)
