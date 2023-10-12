# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from pytest import fixture

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
