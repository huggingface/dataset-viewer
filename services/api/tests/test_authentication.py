from typing import Dict

import pytest
import responses
from starlette.requests import Headers, Request

from api.authentication import auth_check
from api.utils import ExternalAuthenticatedError, ExternalUnauthenticatedError

from .utils import request_callback


def test_no_auth_check() -> None:
    assert auth_check("dataset") is True


def test_invalid_auth_check_url() -> None:
    with pytest.raises(ValueError):
        auth_check("dataset", external_auth_url="https://auth.check/")


@responses.activate
def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(RuntimeError):
        auth_check("dataset", external_auth_url="https://auth.check/%s")


@responses.activate
def test_external_auth_responses_without_request() -> None:
    dataset = "dataset"
    url = "https://auth.check/%s"
    responses.add(responses.GET, url % dataset, status=200)
    assert auth_check(dataset, external_auth_url=url) is True

    responses.add(responses.GET, url % dataset, status=401)
    with pytest.raises(ExternalUnauthenticatedError):
        auth_check(dataset, external_auth_url=url)

    responses.add(responses.GET, url % dataset, status=403)
    with pytest.raises(ExternalAuthenticatedError):
        auth_check(dataset, external_auth_url=url)

    responses.add(responses.GET, url % dataset, status=404)
    with pytest.raises(ValueError):
        auth_check(dataset, external_auth_url=url)


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
    dataset = "dataset"
    url = "https://auth.check/%s"

    responses.add_callback(responses.GET, url % dataset, callback=request_callback)

    with pytest.raises(ExternalUnauthenticatedError):
        auth_check(
            dataset,
            external_auth_url=url,
            request=create_request(headers={"cookie": "some cookie"}),
        )

    with pytest.raises(ExternalAuthenticatedError):
        auth_check(
            dataset,
            external_auth_url=url,
            request=create_request(headers={"authorization": "Bearer token"}),
        )

    assert (
        auth_check(
            dataset,
            external_auth_url=url,
            request=create_request(headers={}),
        )
        is True
    )
