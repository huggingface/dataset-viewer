# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from pathlib import Path

from environs import Env
from pytest import fixture

from libcommon.storage import StrPath, init_cached_assets_dir

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.datasets"]


@fixture(scope="session")
def env() -> Env:
    return Env(expand_vars=True)


@fixture(scope="session")
def cache_mongo_host(env: Env) -> str:
    try:
        url = env.str(name="CACHE_MONGO_URL")
        if type(url) is not str:
            raise ValueError("CACHE_MONGO_URL is not set")
        return url
    except Exception as e:
        raise ValueError("CACHE_MONGO_URL is not set") from e


@fixture(scope="session")
def queue_mongo_host(env: Env) -> str:
    try:
        url = env.str(name="QUEUE_MONGO_URL")
        if type(url) is not str:
            raise ValueError("QUEUE_MONGO_URL is not set")
        return url
    except Exception as e:
        raise ValueError("QUEUE_MONGO_URL is not set") from e


@fixture
def cached_assets_directory(tmp_path: Path) -> StrPath:
    cached_assets_directory = tmp_path / "cached-assets"
    return init_cached_assets_dir(cached_assets_directory)
