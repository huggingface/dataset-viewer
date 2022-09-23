# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

import pytest

port = 8888
host = "localhost"
HF_ENDPOINT = f"http://{host}:{port}"
HF_AUTH_PATH = "/api/datasets/%s/auth-check"

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_AUTH_PATH"] = HF_AUTH_PATH


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return (host, 8888)


@pytest.fixture(scope="session")
def hf_endpoint():
    return HF_ENDPOINT


@pytest.fixture(scope="session")
def hf_auth_path():
    return HF_AUTH_PATH
