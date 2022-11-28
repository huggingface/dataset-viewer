# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from .utils import poll


@pytest.mark.parametrize("endpoint", ["/", "/healthcheck", "/metrics"])
def test_healthcheck(endpoint: str) -> None:
    # this tests ensures the /healthcheck and the /metrics endpoints are hidden
    response = poll(endpoint, expected_code=404)
    assert response.status_code == 404, f"{response.status_code} - {response.text}"
    assert "Not Found" in response.text, response.text
