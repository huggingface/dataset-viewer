# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .utils import SEARCH_UVICORN_PORT, poll


def test_healthcheck() -> None:
    # this tests ensures the /healthcheck and the /metrics endpoints are hidden
    response = poll("/healthcheck", expected_code=200, url=SEARCH_UVICORN_PORT)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    assert "ok" in response.text, response.text
