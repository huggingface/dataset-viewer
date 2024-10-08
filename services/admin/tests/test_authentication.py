# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Optional

import pytest
from libapi.exceptions import ExternalAuthenticatedError, ExternalUnauthenticatedError
from pytest_httpx import HTTPXMock
from starlette.datastructures import Headers
from starlette.requests import Request

from admin.authentication import auth_check

from .utils import request_callback

pytestmark = pytest.mark.anyio


async def test_no_auth_check() -> None:
    assert await auth_check()


async def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(RuntimeError):
        await auth_check(external_auth_url="https://auth.check", organization="org")


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
async def test_external_auth_responses_without_request(
    status: int, error: Optional[type[Exception]], httpx_mock: HTTPXMock
) -> None:
    url = "https://auth.check"
    body = '{"orgs": [{"name": "org1"}], "auth": {"type": "access_token", "accessToken": {"fineGrained": {"scoped": [{"entity": {"type": "org", "name": "org1"}, "permissions": ["repo.write"]}]}}}}'
    httpx_mock.add_response(method="GET", url=url, status_code=status, text=body)
    if error is None:
        assert await auth_check(external_auth_url=url, organization="org1")
    else:
        with pytest.raises(error):
            await auth_check(external_auth_url=url, organization="org1")


@pytest.mark.parametrize(
    "org,status,error",
    [("org1", 200, None), ("org2", 403, ExternalAuthenticatedError)],
)
async def test_org(org: str, status: int, error: Optional[type[Exception]], httpx_mock: HTTPXMock) -> None:
    url = "https://auth.check"
    body = '{"orgs": [{"name": "org1"}], "auth": {"type": "access_token", "accessToken": {"fineGrained": {"scoped": [{"entity": {"type": "org", "name": "org1"}, "permissions": ["repo.write"]}]}}}}'
    httpx_mock.add_response(method="GET", url=url, status_code=status, text=body)
    if error is None:
        assert await auth_check(external_auth_url=url, organization=org)
    else:
        with pytest.raises(error):
            await auth_check(external_auth_url=url, organization=org)


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


async def test_valid_responses_with_request(httpx_mock: HTTPXMock) -> None:
    url = "https://auth.check"
    organization = "org1"

    httpx_mock.add_callback(method="GET", url=url, callback=request_callback)

    with pytest.raises(ExternalAuthenticatedError):
        await auth_check(
            external_auth_url=url,
            request=create_request(headers={"authorization": "Bearer token"}),
            organization=organization,
        )

    assert await auth_check(
        external_auth_url=url,
        request=create_request(headers={}),
        organization=organization,
    )
