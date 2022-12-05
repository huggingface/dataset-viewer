# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import shutil
from typing import Iterator

import datasets.config
from pytest import MonkeyPatch, TempPathFactory, fixture

from datasets_based.config import AppConfig

from .fixtures.hub import HubDatasets

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets", "tests.fixtures.files", "tests.fixtures.hub"]


@fixture(scope="session")
def datasets_cache_directory(tmp_path_factory: TempPathFactory) -> None:
    datasets.config.HF_DATASETS_CACHE = tmp_path_factory.mktemp("data")


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session(
    hf_endpoint: str,
    app_hf_token: str,
    user_hf_token: str,
    hub_datasets: HubDatasets,
    datasets_cache_directory: None,
) -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", hf_endpoint)
    monkeypatch_session.setenv("COMMON_HF_TOKEN", app_hf_token)
    monkeypatch_session.setenv("COMMON_ASSETS_BASE_URL", "http://localhost/assets")
    monkeypatch_session.setenv("FIRST_ROWS_MAX_NUMBER", "7")
    monkeypatch_session.setenv("PARQUET_SUPPORTED_DATASETS", ",".join([d["name"] for d in hub_datasets.values()]))
    monkeypatch_session.setenv("PARQUET_COMMITTER_HF_TOKEN", user_hf_token)
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session", autouse=True)
def app_config(monkeypatch_session: MonkeyPatch) -> AppConfig:
    app_config = AppConfig()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


def _clean_datasets_cache() -> None:
    datasets_cache_directory = datasets.config.HF_DATASETS_CACHE
    print(f"DATASETS CACHE DIR - path: {datasets_cache_directory}")
    print(f"DATASETS CACHE DIR - contents before deleting: {os.listdir(datasets_cache_directory)}")
    shutil.rmtree(datasets_cache_directory, ignore_errors=True)
    os.mkdir(datasets_cache_directory)
    print(f"DATASETS CACHE DIR - contents after deleting: {os.listdir(datasets_cache_directory)}")
