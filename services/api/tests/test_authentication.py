# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import time
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, Mapping, Optional

import jwt
import pytest
import werkzeug.wrappers
from pytest_httpserver import HTTPServer
from starlette.datastructures import Headers
from starlette.requests import Request
from werkzeug.wrappers import Request as WerkzeugRequest
from werkzeug.wrappers import Response as WerkzeugResponse

from api.authentication import auth_check
from api.utils import (
    AuthCheckHubRequestError,
    ExternalAuthenticatedError,
    ExternalUnauthenticatedError,
)

from .test_jwt_token import (
    algorithm_rs256,
    dataset_ok,
    payload_ok,
    private_key,
    public_key,
)
from .utils import auth_callback


def test_no_auth_check() -> None:
    assert auth_check("dataset")


def test_invalid_auth_check_url() -> None:
    with pytest.raises(ValueError):
        auth_check("dataset", external_auth_url="https://auth.check/")


def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(AuthCheckHubRequestError):
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


TIMEOUT_TIME = 0.2


def sleeping(_: werkzeug.wrappers.Request) -> werkzeug.wrappers.Response:
    time.sleep(TIMEOUT_TIME)
    return werkzeug.wrappers.Response(status=200)


@pytest.mark.parametrize(
    "hf_timeout_seconds,expectation",
    [
        (TIMEOUT_TIME * 2, does_not_raise()),
        (None, does_not_raise()),
        (TIMEOUT_TIME / 2, pytest.raises(AuthCheckHubRequestError)),
    ],
)
def test_hf_timeout_seconds(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    hf_timeout_seconds: Optional[float],
    expectation: Any,
) -> None:
    dataset = "dataset"
    external_auth_url = hf_endpoint + hf_auth_path
    httpserver.expect_request(hf_auth_path % dataset).respond_with_handler(func=sleeping)
    with expectation:
        auth_check(dataset, external_auth_url=external_auth_url, hf_timeout_seconds=hf_timeout_seconds)


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


def raise_value_error(request: WerkzeugRequest) -> WerkzeugResponse:
    return WerkzeugResponse(status=500)  # <- will raise ValueError in auth_check


@pytest.mark.parametrize(
    "hf_jwt_public_key,header,payload,expectation",
    [
        (None, None, payload_ok, pytest.raises(ValueError)),
        (None, "X-Api-Key", payload_ok, pytest.raises(ValueError)),
        (public_key, None, payload_ok, pytest.raises(ValueError)),
        (public_key, "X-Api-Key", {}, pytest.raises(ValueError)),
        (public_key, "X-Api-Key", payload_ok, does_not_raise()),
        (public_key, "x-api-key", payload_ok, does_not_raise()),
    ],
)
def test_bypass_auth_public_key(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    hf_jwt_public_key: Optional[str],
    header: Optional[str],
    payload: Dict[str, str],
    expectation: Any,
) -> None:
    external_auth_url = hf_endpoint + hf_auth_path
    httpserver.expect_request(hf_auth_path % dataset_ok).respond_with_handler(raise_value_error)
    headers = {header: jwt.encode(payload, private_key, algorithm=algorithm_rs256)} if header else {}
    with expectation:
        auth_check(
            dataset=dataset_ok,
            external_auth_url=external_auth_url,
            request=create_request(headers=headers),
            hf_jwt_public_key=hf_jwt_public_key,
            hf_jwt_algorithm=algorithm_rs256,
        )
