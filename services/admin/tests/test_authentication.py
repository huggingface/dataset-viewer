# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Mapping, Optional, Type

import pytest
import responses
from starlette.requests import Headers, Request

from admin.authentication import auth_check
from admin.utils import ExternalAuthenticatedError, ExternalUnauthenticatedError

from .utils import request_callback


def test_no_auth_check() -> None:
    assert auth_check()


@responses.activate
def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(RuntimeError):
        auth_check(external_auth_url="https://auth.check", organization="org")


@responses.activate
@pytest.mark.parametrize(
    "status,error",
    [
        (200, None),
        (401, ExternalUnauthenticatedError),
        (403, ExternalAuthenticatedError),
        (404, ExternalAuthenticatedError),
        (429, ValueError),
    ],
)
def test_external_auth_responses_without_request(status: int, error: Optional[Type[Exception]]) -> None:
    url = "https://auth.check"
    body = '{"orgs": [{"name": "org1"}]}'
    responses.add(responses.GET, url, status=status, body=body)
    if error is None:
        assert auth_check(external_auth_url=url, organization="org1")
    else:
        with pytest.raises(error):
            auth_check(external_auth_url=url, organization="org1")


@responses.activate
@pytest.mark.parametrize(
    "org,status,error",
    [("org1", 200, None), ("org2", 403, ExternalAuthenticatedError)],
)
def test_org(org: str, status: int, error: Optional[Type[Exception]]) -> None:
    url = "https://auth.check"
    body = '{"orgs": [{"name": "org1"}]}'
    responses.add(responses.GET, url, status=status, body=body)
    if error is None:
        assert auth_check(external_auth_url=url, organization=org)
    else:
        with pytest.raises(error):
            auth_check(external_auth_url=url, organization=org)


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


@responses.activate
def test_valid_responses_with_request() -> None:
    url = "https://auth.check"
    organization = "org1"

    responses.add_callback(responses.GET, url, callback=request_callback)

    with pytest.raises(ExternalAuthenticatedError):
        auth_check(
            external_auth_url=url,
            request=create_request(headers={"authorization": "Bearer token"}),
            organization=organization,
        )

    assert auth_check(
        external_auth_url=url,
        request=create_request(headers={}),
        organization=organization,
    )
