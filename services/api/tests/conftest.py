# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from api.config import AppConfig, UvicornConfig


@pytest.fixture(scope="session")
def app_config():
    app_config = AppConfig()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


@pytest.fixture(scope="session")
def uvicorn_config():
    return UvicornConfig()


@pytest.fixture(scope="session")
def httpserver_listen_address(uvicorn_config: UvicornConfig):
    return (uvicorn_config.hostname, uvicorn_config.port)


@pytest.fixture(scope="session")
def hf_endpoint(app_config: AppConfig):
    return app_config.common.hf_endpoint


@pytest.fixture(scope="session")
def hf_auth_path(app_config: AppConfig):
    return app_config.api.hf_auth_path
