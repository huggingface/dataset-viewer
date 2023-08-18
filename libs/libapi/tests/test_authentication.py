# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datetime
import time
from contextlib import nullcontext as does_not_raise
from typing import Any, Mapping, Optional

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
exp_ok = datetime.datetime.now().timestamp() + 1000
read_ok = True
algorithm_rs256 = "RS256"

dataset_public = "dataset_public"
dataset_protected_with_access = "dataset_protected_with_access"
dataset_protected_without_access = "dataset_protected_without_access"
dataset_inexistent = "dataset_inexistent"
dataset_throttled = "dataset_throttled"

cookie_ok = "cookie ok"
cookie_wrong = "cookie wrong"
api_token_ok = "api token ok"
api_token_wrong = "api token wrong"


def auth_callback(request: WerkzeugRequest) -> WerkzeugResponse:
    """Simulates the https://huggingface.co/api/datasets/%s/auth-check Hub API endpoint.

    It returns:
    - 200: if the user can access the dataset
    - 401: if the user is not authenticated
    - 403: if the user is authenticated but can't access the dataset
    - 404: if the user is authenticated but the dataset doesn't exist
    - 429: if the user is authenticated but the request is throttled

    Args:
        request (WerkzeugRequest): the request sent to the endpoint

    Returns:
        WerkzeugResponse: the response sent by the endpoint
    """
    dataset = request.path.split("/")[-2]
    if dataset == dataset_public:
        # a public dataset always has read access
        return WerkzeugResponse(status=200)
    if request.headers.get("cookie") != cookie_ok and request.headers.get("authorization") != f"Bearer {api_token_ok}":
        # the user is not authenticated
        return WerkzeugResponse(status=401)
    if dataset == dataset_protected_with_access:
        # the user is authenticated and has access to the dataset
        return WerkzeugResponse(status=200)
    if dataset == dataset_protected_without_access:
        # the user is authenticated but doesn't have access to the dataset
        return WerkzeugResponse(status=403)
    if dataset == dataset_inexistent:
        # the user is authenticated but the dataset doesn't exist
        return WerkzeugResponse(status=404)
    if dataset == dataset_throttled:
        # the user is authenticated but the request is throttled (too many requests)
        return WerkzeugResponse(status=429)
    raise RuntimeError(f"Unexpected dataset: {dataset}")


def test_no_external_auth_check() -> None:
    assert auth_check(dataset_public)


def test_invalid_external_auth_check_url() -> None:
    with pytest.raises(ValueError):
        auth_check(dataset_public, external_auth_url="https://doesnotexist/")


def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(AuthCheckHubRequestError):
        auth_check(dataset_public, external_auth_url="https://doesnotexist/%s")


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


def get_jwt(dataset: str) -> str:
    return jwt.encode(
        {"sub": f"datasets/{dataset}", "read": read_ok, "exp": exp_ok}, private_key, algorithm=algorithm_rs256
    )


def assert_auth_headers(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    dataset: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    external_auth_url = hf_endpoint + hf_auth_path
    httpserver.expect_request(hf_auth_path % dataset).respond_with_handler(auth_callback)
    with expectation:
        auth_check(
            dataset,
            external_auth_url=external_auth_url,
            request=create_request(headers=headers),
            hf_jwt_public_key=public_key,
            hf_jwt_algorithm=algorithm_rs256,
        )


@pytest.mark.parametrize(
    "headers,expectation",
    [
        ({}, does_not_raise()),
        ({"Authorization": f"Bearer {api_token_wrong}"}, does_not_raise()),
        ({"Authorization": api_token_ok}, does_not_raise()),
        ({"Cookie": cookie_wrong}, does_not_raise()),
        ({"Authorization": f"Bearer {api_token_ok}"}, does_not_raise()),
        ({"Cookie": cookie_ok}, does_not_raise()),
        ({"X-Api-Key": get_jwt(dataset_public)}, does_not_raise()),
        ({"Authorization": f"Bearer {get_jwt(dataset_public)}"}, does_not_raise()),
    ],
)
def test_external_auth_service_dataset_public(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    assert_auth_headers(httpserver, hf_endpoint, hf_auth_path, dataset_public, headers, expectation)


@pytest.mark.parametrize(
    "headers,expectation",
    [
        ({}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_wrong}"}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": api_token_ok}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Cookie": cookie_wrong}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_ok}"}, does_not_raise()),
        ({"Cookie": cookie_ok}, does_not_raise()),
        ({"X-Api-Key": get_jwt(dataset_protected_with_access)}, does_not_raise()),
        ({"Authorization": f"Bearer jwt:{get_jwt(dataset_protected_with_access)}"}, does_not_raise()),
    ],
)
def test_external_auth_service_dataset_protected_with_access(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    assert_auth_headers(httpserver, hf_endpoint, hf_auth_path, dataset_protected_with_access, headers, expectation)


@pytest.mark.parametrize(
    "headers,expectation",
    [
        ({}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_wrong}"}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": api_token_ok}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Cookie": cookie_wrong}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_ok}"}, pytest.raises(ExternalAuthenticatedError)),
        ({"Cookie": cookie_ok}, pytest.raises(ExternalAuthenticatedError)),
        ({"X-Api-Key": get_jwt(dataset_protected_without_access)}, does_not_raise()),
        ({"Authorization": f"Bearer jwt:{get_jwt(dataset_protected_without_access)}"}, does_not_raise()),
    ],
)
def test_external_auth_service_dataset_protected_without_access(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    assert_auth_headers(httpserver, hf_endpoint, hf_auth_path, dataset_protected_without_access, headers, expectation)


@pytest.mark.parametrize(
    "headers,expectation",
    [
        ({}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_wrong}"}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": api_token_ok}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Cookie": cookie_wrong}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_ok}"}, pytest.raises(ExternalAuthenticatedError)),
        ({"Cookie": cookie_ok}, pytest.raises(ExternalAuthenticatedError)),
        ({"X-Api-Key": get_jwt(dataset_inexistent)}, does_not_raise()),
        ({"Authorization": f"Bearer jwt:{get_jwt(dataset_inexistent)}"}, does_not_raise()),
    ],
)
def test_external_auth_service_dataset_inexistent(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    assert_auth_headers(httpserver, hf_endpoint, hf_auth_path, dataset_inexistent, headers, expectation)


@pytest.mark.parametrize(
    "headers,expectation",
    [
        ({}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_wrong}"}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": api_token_ok}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Cookie": cookie_wrong}, pytest.raises(ExternalUnauthenticatedError)),
        ({"Authorization": f"Bearer {api_token_ok}"}, pytest.raises(ValueError)),
        ({"Cookie": cookie_ok}, pytest.raises(ValueError)),
        ({"X-Api-Key": get_jwt(dataset_throttled)}, does_not_raise()),
        ({"Authorization": f"Bearer jwt:{get_jwt(dataset_throttled)}"}, does_not_raise()),
    ],
)
def test_external_auth_service_dataset_throttled(
    httpserver: HTTPServer,
    hf_endpoint: str,
    hf_auth_path: str,
    headers: Mapping[str, str],
    expectation: Any,
) -> None:
    assert_auth_headers(httpserver, hf_endpoint, hf_auth_path, dataset_throttled, headers, expectation)
