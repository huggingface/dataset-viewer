# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from contextlib import nullcontext as does_not_raise
from typing import Any, Mapping

import pytest
from pytest_httpserver import HTTPServer
from starlette.requests import Headers, Request

from api.authentication import auth_check
from api.utils import ExternalAuthenticatedError, ExternalUnauthenticatedError

from .utils import auth_callback


def test_no_auth_check() -> None:
    assert auth_check("dataset") is True


def test_invalid_auth_check_url() -> None:
    with pytest.raises(ValueError):
        auth_check("dataset", external_auth_url="https://auth.check/")


def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(RuntimeError):
        auth_check("dataset", external_auth_url="https://auth.check/%s")


@pytest.mark.parametrize(
    "status_code,expectation",
    [
        (200, does_not_raise()),
        (401, pytest.raises(ExternalUnauthenticatedError)),
        (403, pytest.raises(ExternalAuthenticatedError)),
        (404, pytest.raises(ExternalAuthenticatedError)),
        (429, pytest.raises(ValueError)),
    ],
)
def test_external_auth_responses_without_request(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    status_code: int,
    expectation: Any,
) -> None:
    dataset = "dataset"
    external_auth_url = hf_endpoint + hf_auth_path
    httpserver.expect_request(hf_auth_path % dataset).respond_with_data(status=status_code)
    with expectation:
        auth_check(dataset, external_auth_url=external_auth_url)


def create_request(headers: Mapping[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "path": "/some-path",
            "headers": Headers(headers).raw,
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "client": ("127.0.0.1", 8080),
            "server": ("some.server", 443),
        }
    )


@pytest.mark.parametrize(
    "headers,expectation",
    [
        ({"Cookie": "some cookie"}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": "Bearer invalid"}, pytest.raises(ExternalAuthenticatedError)),
        ({}, does_not_raise()),
    ],
)
def test_valid_responses_with_request(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    dataset = "dataset"
    external_auth_url = hf_endpoint + hf_auth_path
    httpserver.expect_request(hf_auth_path % dataset).respond_with_handler(auth_callback)
    with expectation:
        auth_check(
            dataset,
            external_auth_url=external_auth_url,
            request=create_request(headers=headers),
        )
