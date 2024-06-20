# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Iterator
from pathlib import Path

from libcommon.config import CacheConfig, QueueConfig
from libcommon.queue.utils import _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database
from pytest import MonkeyPatch, TempPathFactory, fixture

from libapi.config import ApiConfig


@fixture(scope="session")
def hostname() -> str:
    return "localhost"


@fixture(scope="session")
def port() -> str:
    return "8888"


@fixture(scope="session")
def httpserver_listen_address(hostname: str, port: int) -> tuple[str, int]:
    return (hostname, port)


@fixture(scope="session")
def hf_endpoint(hostname: str, port: int) -> str:
    return f"http://{hostname}:{port}"


@fixture(scope="session")
def api_config(hf_endpoint: str) -> ApiConfig:
    return ApiConfig.from_env(hf_endpoint=hf_endpoint)


@fixture(scope="session")
def hf_auth_path(api_config: ApiConfig) -> str:
    return api_config.hf_auth_path


@fixture
def anyio_backend() -> str:
    return "asyncio"


@fixture
def image_path() -> str:
    image_path = Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"
    assert image_path.is_file()
    return str(image_path)


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session(tmp_path_factory: TempPathFactory) -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "dataset_viewer_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "dataset_viewer_queue_test")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(autouse=True)
def cache_mongo_resource(monkeypatch_session: MonkeyPatch) -> Iterator[CacheMongoResource]:
    cache_config = CacheConfig.from_env()
    if "test" not in cache_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    with CacheMongoResource(database=cache_config.mongo_database, host=cache_config.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture(autouse=True)
def queue_mongo_resource(monkeypatch_session: MonkeyPatch) -> Iterator[QueueMongoResource]:
    queue_config = QueueConfig.from_env()
    if "test" not in queue_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    with QueueMongoResource(database=queue_config.mongo_database, host=queue_config.mongo_url) as resource:
        yield resource
        _clean_queue_database()
