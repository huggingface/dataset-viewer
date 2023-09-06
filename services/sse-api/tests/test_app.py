# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from starlette.testclient import TestClient

from sse_api.app import create_app_with_config
from sse_api.config import AppConfig


@pytest.fixture(scope="module")
def client(monkeypatch_session: pytest.MonkeyPatch, app_config: AppConfig) -> TestClient:
    return TestClient(create_app_with_config(app_config=app_config))


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"
