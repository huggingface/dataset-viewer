# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json

import pytest
from pytest_httpserver import HTTPServer

from api.dataset import UnsupportedDatasetError, check_support


@pytest.mark.parametrize(
    "private,exists,raises",
    [(True, False, True), (False, False, False), (True, False, True)],
)
def test_check_supported(httpserver: HTTPServer, hf_endpoint: str, private: bool, exists: bool, raises: bool) -> None:
    dataset = "dataset"
    endpoint = f"/api/datasets/{dataset}"
    hf_token = "dummy_token"

    headers = None if exists else {"X-Error-Code": "RepoNotFound"}
    httpserver.expect_request(endpoint).respond_with_data(json.dumps({"private": private}), headers=headers)
    if raises:
        with pytest.raises(UnsupportedDatasetError):
            check_support(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
    else:
        check_support(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
