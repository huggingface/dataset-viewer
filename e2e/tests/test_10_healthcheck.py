# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from .utils import poll


@pytest.mark.parametrize("endpoint", ["/healthcheck", "/admin/healthcheck"])
def test_healthcheck(endpoint: str) -> None:
    # this tests ensures the /healthcheck are accessible
    response = poll(endpoint, expected_code=200)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    assert "ok" in response.text, response.text


@pytest.mark.parametrize("endpoint", ["/", "/metrics", "/admin/metrics"])
def test_hidden(endpoint: str) -> None:
    # this tests ensures the root / and the /metrics endpoints are hidden
    response = poll(endpoint, expected_code=404)
    assert response.status_code == 404, f"{response.status_code} - {response.text}"
    assert "Not Found" in response.text, response.text
