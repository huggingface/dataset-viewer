# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path
from typing import Iterator

from pytest import MonkeyPatch, fixture

import datasets.config
from datasets_based.config import AppConfig
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import _clean_cache_database

from .constants import CI_APP_TOKEN, CI_HUB_ENDPOINT, CI_URL_TEMPLATE, CI_USER_TOKEN


@fixture
def datasets_cache_directory(tmp_path: Path) -> Path:
    return tmp_path


@fixture
def modules_cache_directory(tmp_path: Path) -> Path:
    return tmp_path


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
def set_env_vars() -> Iterator[MonkeyPatch]:
    mp = MonkeyPatch()
    mp.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    mp.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    mp.setenv("COMMON_HF_ENDPOINT", CI_HUB_ENDPOINT)
    mp.setenv("COMMON_HF_TOKEN", CI_APP_TOKEN)
    mp.setenv("COMMON_ASSETS_BASE_URL", "http://localhost/assets")
    mp.setenv("FIRST_ROWS_MAX_NUMBER", "7")
    mp.setenv("PARQUET_MAX_DATASET_SIZE", "10_000")
    mp.setenv("PARQUET_COMMITTER_HF_TOKEN", CI_USER_TOKEN)
    yield mp
    mp.undo()


@fixture
def set_datasets_dirs(
    datasets_cache_directory: Path,
    modules_cache_directory: Path,
) -> None:
    datasets.config.HF_DATASETS_CACHE = datasets_cache_directory
    datasets.config.DOWNLOADED_DATASETS_PATH = (
        datasets.config.HF_DATASETS_CACHE / datasets.config.DOWNLOADED_DATASETS_DIR
    )
    datasets.config.EXTRACTED_DATASETS_PATH = (
        datasets.config.HF_DATASETS_CACHE / datasets.config.EXTRACTED_DATASETS_DIR
    )
    datasets.config.HF_MODULES_CACHE = modules_cache_directory


@fixture
def app_config(set_env_vars: MonkeyPatch, set_datasets_dirs: None) -> Iterator[AppConfig]:
    app_config = AppConfig()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    yield app_config
    # Clean the database after each test. Must be done in test databases only, ensured by the check above!
    # TODO: use a parameter to pass a reference to the database, instead of relying on the implicit global variable
    # managed by mongoengine
    _clean_cache_database()
    _clean_queue_database()


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]
