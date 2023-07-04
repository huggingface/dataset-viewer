# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
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

from libapi.authentication import auth_check
from libapi.exceptions import (
    AuthCheckHubRequestError,
    ExternalAuthenticatedError,
    ExternalUnauthenticatedError,
)

private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIBOQIBAAJAZTmplhS/Jd73ycVut7TglMObheQqXM7RZYlwazLU4wpfIVIwOh9I
sCZGSgLyFq42KWIikKLEs/yqx3pRGfq+rwIDAQABAkAMyF9WCICq86Eu5bO5lynV
H26AVfPTjHp87AI6R00C7p9n8hO/DhHaHpc3InOSsXsw9d2hmz37jwwBFiwMHMMh
AiEAtbttHlIO+yO29oXw4P6+yO11lMy1UpT1sPVTnR9TXbUCIQCOl7Zuyy2ZY9ZW
pDhW91x/14uXjnLXPypgY9bcfggJUwIhAJQG1LzrzjQWRUPMmgZKuhBkC3BmxhM8
LlwzmCXVjEw5AiA7JnAFEb9+q82T71d3q/DxD0bWvb6hz5ASoBfXK2jGBQIgbaQp
h4Tk6UJuj1xgKNs75Pk3pG2tj8AQiuBk3l62vRU=
-----END RSA PRIVATE KEY-----"""
public_key = """-----BEGIN PUBLIC KEY-----
MFswDQYJKoZIhvcNAQEBBQADSgAwRwJAZTmplhS/Jd73ycVut7TglMObheQqXM7R
ZYlwazLU4wpfIVIwOh9IsCZGSgLyFq42KWIikKLEs/yqx3pRGfq+rwIDAQAB
-----END PUBLIC KEY-----"""

dataset_ok = "dataset"
exp_ok = datetime.datetime.now().timestamp() + 1000
read_ok = True
sub_ok = f"datasets/{dataset_ok}"
payload_ok = {"sub": sub_ok, "read": read_ok, "exp": exp_ok}
algorithm_rs256 = "RS256"


def auth_callback(request: WerkzeugRequest) -> WerkzeugResponse:
    # return 401 if a cookie has been provided, 404 if a token has been provided,
    # and 200 if none has been provided
    #
    # caveat: the returned status codes don't simulate the reality
    # they're just used to check every case
    return WerkzeugResponse(
        status=401 if request.headers.get("cookie") else 404 if request.headers.get("authorization") else 200
    )


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
