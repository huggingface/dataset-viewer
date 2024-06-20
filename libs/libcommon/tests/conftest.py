# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from collections.abc import Iterator

from environs import Env
from pytest import TempPathFactory, fixture

from libcommon.config import ParquetMetadataConfig
from libcommon.new_queue.utils import _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database
from libcommon.storage import StrPath, init_parquet_metadata_dir
from libcommon.storage_client import StorageClient

from .constants import ASSETS_BASE_URL

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.fsspec"]


@fixture(scope="session")
def env() -> Env:
    return Env(expand_vars=True)


@fixture(scope="session")
def cache_mongo_host(env: Env) -> str:
    try:
        url = env.str(name="CACHE_MONGO_URL")
        if not isinstance(url, str):
            raise ValueError("CACHE_MONGO_URL is not set")
        return url
    except Exception as e:
        raise ValueError("CACHE_MONGO_URL is not set") from e


@fixture(scope="session")
def queue_mongo_host(env: Env) -> str:
    try:
        url = env.str(name="QUEUE_MONGO_URL")
        if not isinstance(url, str):
            raise ValueError("QUEUE_MONGO_URL is not set")
        return url
    except Exception as e:
        raise ValueError("QUEUE_MONGO_URL is not set") from e


@fixture
def queue_mongo_resource(queue_mongo_host: str) -> Iterator[QueueMongoResource]:
    database = "dataset_viewer_queue_test"
    host = queue_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with QueueMongoResource(database=database, host=host, server_selection_timeout_ms=3_000) as queue_mongo_resource:
        if not queue_mongo_resource.is_available():
            raise RuntimeError("Mongo resource is not available")
        yield queue_mongo_resource
        _clean_queue_database()
        queue_mongo_resource.release()


@fixture
def cache_mongo_resource(cache_mongo_host: str) -> Iterator[CacheMongoResource]:
    database = "dataset_viewer_cache_test"
    host = cache_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with CacheMongoResource(database=database, host=host) as cache_mongo_resource:
        yield cache_mongo_resource
        _clean_cache_database()
        cache_mongo_resource.release()


@fixture
def parquet_metadata_directory() -> StrPath:
    return init_parquet_metadata_dir(ParquetMetadataConfig().storage_directory)


@fixture(scope="session")
def storage_client(tmp_path_factory: TempPathFactory) -> StorageClient:
    return StorageClient(
        protocol="file", storage_root=str(tmp_path_factory.getbasetemp()), base_url=ASSETS_BASE_URL, overwrite=True
    )
