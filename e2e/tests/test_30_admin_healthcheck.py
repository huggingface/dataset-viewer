# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from .utils import ADMIN_URL, poll


def test_healthcheck() -> None:
    response = poll("/healthcheck", expected_code=200, url=ADMIN_URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    assert "ok" in response.text, response.text
