# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json

import pytest
from pytest_httpserver import HTTPServer

from api.dataset import is_supported


@pytest.mark.parametrize(
    "private,exists,expected",
    [(True, False, False), (False, False, True), (True, False, False)],
)
def test_is_supported(httpserver: HTTPServer, hf_endpoint: str, private: bool, exists: bool, expected: bool) -> None:
    dataset = "dataset"
    endpoint = f"/api/datasets/{dataset}"
    hf_token = "dummy_token"

    headers = None if exists else {"X-Error-Code": "RepoNotFound"}
    httpserver.expect_request(endpoint).respond_with_data(json.dumps({"private": private}), headers=headers)
    assert is_supported(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token) is expected
