# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path
from typing import Iterator

from libcommon.config import CacheConfig, QueueConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import _clean_queue_database
from libcommon.resources import (
    AssetsDirectoryResource,
    CacheDatabaseResource,
    QueueDatabaseResource,
)
from libcommon.simple_cache import _clean_cache_database
from pytest import MonkeyPatch, fixture

from datasets_based.config import AppConfig, FirstRowsConfig
from datasets_based.resources import LibrariesResource

from .constants import CI_APP_TOKEN, CI_HUB_ENDPOINT, CI_URL_TEMPLATE, CI_USER_TOKEN


@fixture
def datasets_cache_directory(tmp_path: Path) -> Path:
    return tmp_path / "datasets"


@fixture
def modules_cache_directory(tmp_path: Path) -> Path:
    return tmp_path / "modules"


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session", autouse=True)
def monkeypatch_session() -> Iterator[MonkeyPatch]:
    mp = MonkeyPatch()
    mp.setattr("huggingface_hub.file_download.HUGGINGFACE_CO_URL_TEMPLATE", CI_URL_TEMPLATE)
    # ^ see https://github.com/huggingface/datasets/pull/5196#issuecomment-1322191056
    mp.setattr("datasets.config.HF_ENDPOINT", CI_HUB_ENDPOINT)
    mp.setattr("datasets.config.HF_UPDATE_DOWNLOAD_COUNTS", False)
    yield mp
    mp.undo()


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture
def set_env_vars(datasets_cache_directory: Path, modules_cache_directory: Path) -> Iterator[MonkeyPatch]:
    mp = MonkeyPatch()
    mp.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    mp.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    mp.setenv("COMMON_HF_ENDPOINT", CI_HUB_ENDPOINT)
    mp.setenv("COMMON_HF_TOKEN", CI_APP_TOKEN)
    mp.setenv("ASSETS_BASE_URL", "http://localhost/assets")
    mp.setenv("FIRST_ROWS_MAX_NUMBER", "7")
    mp.setenv("PARQUET_AND_DATASET_INFO_MAX_DATASET_SIZE", "10_000")
    mp.setenv("PARQUET_AND_DATASET_INFO_COMMITTER_HF_TOKEN", CI_USER_TOKEN)
    mp.setenv("DATASETS_BASED_HF_DATASETS_CACHE", str(datasets_cache_directory))
    mp.setenv("HF_MODULES_CACHE", str(modules_cache_directory))
    yield mp
    mp.undo()


@fixture
def app_config(set_env_vars: MonkeyPatch) -> Iterator[AppConfig]:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    yield app_config


@fixture
def cache_database_resource(app_config: AppConfig) -> Iterator[CacheDatabaseResource]:
    with CacheDatabaseResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture
def queue_database_resource(app_config: AppConfig) -> Iterator[QueueDatabaseResource]:
    with QueueDatabaseResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url) as resource:
        yield resource
        _clean_queue_database()


@fixture
def libraries_resource(app_config: AppConfig) -> Iterator[LibrariesResource]:
    with LibrariesResource(
        common_config=app_config.common,
        datasets_based_config=app_config.datasets_based,
        numba_config=app_config.numba,
    ) as libraries_resource:
        yield libraries_resource


@fixture
def first_rows_config(set_env_vars: MonkeyPatch) -> FirstRowsConfig:
    return FirstRowsConfig.from_env()


@fixture
def assets_directory(first_rows_config: FirstRowsConfig) -> Iterator[AssetsDirectoryResource]:
    with AssetsDirectoryResource(storage_directory=first_rows_config.assets.storage_directory) as resource:
        yield resource.storage_directory


@fixture()
def test_processing_step() -> ProcessingStep:
    return ProcessingStep(
        endpoint="/dummy",
        input_type="dataset",
        requires=None,
        required_by_dataset_viewer=False,
        parent=None,
        ancestors=[],
        children=[],
    )


@fixture()
def cache_config(app_config: AppConfig) -> CacheConfig:
    cache_config = app_config.cache
    if "test" not in cache_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return cache_config


@fixture()
def queue_config(app_config: AppConfig) -> QueueConfig:
    queue_config = app_config.queue
    if "test" not in queue_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return queue_config


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]
