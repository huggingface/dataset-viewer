from typing import Dict

import pytest
import responses
from starlette.requests import Headers, Request

from admin.authentication import auth_check
from admin.utils import ExternalAuthenticatedError, ExternalUnauthenticatedError

from .utils import request_callback


def test_no_auth_check() -> None:
    assert auth_check() is True


@responses.activate
def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(RuntimeError):
        auth_check(external_auth_url="https://auth.check")


@responses.activate
def test_external_auth_responses_without_request() -> None:
    url = "https://auth.check"
    body = '{"orgs": [{"name": "org1"}]}'
    responses.add(responses.GET, url, status=200, body=body)
    assert auth_check(external_auth_url=url, organization=None) is True

    responses.add(responses.GET, url, status=401, body=body)
    with pytest.raises(ExternalUnauthenticatedError):
        auth_check(external_auth_url=url, organization=None)

    responses.add(responses.GET, url, status=403, body=body)
    with pytest.raises(ExternalAuthenticatedError):
        auth_check(external_auth_url=url, organization=None)

    responses.add(responses.GET, url, status=404, body=body)
    with pytest.raises(ExternalAuthenticatedError):
        auth_check(external_auth_url=url, organization=None)

    responses.add(responses.GET, url, status=429, body=body)
    with pytest.raises(ValueError):
        auth_check(external_auth_url=url, organization=None)


@responses.activate
def test_org() -> None:
    url = "https://auth.check"
    body = '{"orgs": [{"name": "org1"}]}'
    responses.add(responses.GET, url, status=200, body=body)
    assert auth_check(external_auth_url=url, organization="org1") is True

    responses.add(responses.GET, url, status=403, body=body)
    with pytest.raises(ExternalAuthenticatedError):
        auth_check(external_auth_url=url, organization="org2")


def create_request(headers: Dict[str, str]) -> Request:
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

    assert (
        auth_check(
            external_auth_url=url,
            request=create_request(headers={}),
            organization=organization,
        )
        is True
    )
