from typing import Dict, Mapping, Tuple, Union

import pytest
import responses
from requests import PreparedRequest
from responses import _Body
from starlette.requests import Headers, Request

from api.authentication import auth_check
from api.utils import (
    ExternalAuthCheckConnectionError,
    ExternalAuthCheckResponseError,
    ExternalAuthCheckUrlFormatError,
    ExternalAuthenticatedError,
    ExternalUnauthenticatedError,
)


def test_no_auth_check() -> None:
    assert auth_check("dataset") is True


def test_invalid_auth_check_url() -> None:
    with pytest.raises(ExternalAuthCheckUrlFormatError):
        auth_check("dataset", external_auth_url="https://huggingface.co/api/datasets/auth-check")


@responses.activate
def test_unreachable_external_auth_check_service() -> None:
    with pytest.raises(ExternalAuthCheckConnectionError):
        auth_check("dataset", external_auth_url="https://huggingface.co/api/datasets/%s/auth-check")


@responses.activate
def test_external_auth_responses_without_request() -> None:
    dataset = "dataset"
    url = "https://huggingface.co/api/datasets/%s/auth-check"
    responses.add(responses.GET, url % dataset, status=200)
    assert auth_check(dataset, external_auth_url=url) is True

    responses.add(responses.GET, url % dataset, status=401)
    with pytest.raises(ExternalUnauthenticatedError):
        auth_check(dataset, external_auth_url=url)

    responses.add(responses.GET, url % dataset, status=403)
    with pytest.raises(ExternalAuthenticatedError):
        auth_check(dataset, external_auth_url=url)

    responses.add(responses.GET, url % dataset, status=404)
    with pytest.raises(ExternalAuthCheckResponseError):
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
    url = "https://huggingface.co/api/datasets/%s/auth-check"

    def request_callback(request: PreparedRequest) -> Union[Exception, Tuple[int, Mapping[str, str], _Body]]:
        # return 200 if a cookie has been provided, 403 if a token has been provided,
        # and 401 if none has been provided
        # there is no logic behind this behavior, it's just to test if the cookie and
        # token are correctly passed to the auth_check service
        if request.headers.get("cookie"):
            return (200, {"Content-Type": "text/plain"}, "OK")
        if request.headers.get("authorization"):
            return (403, {"Content-Type": "text/plain"}, "OK")
        return (401, {"Content-Type": "text/plain"}, "OK")

    responses.add_callback(responses.GET, url % dataset, callback=request_callback)

    with pytest.raises(ExternalUnauthenticatedError):
        auth_check(
            dataset,
            external_auth_url=url,
            request=create_request(headers={}),
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
            request=create_request(headers={"cookie": "some cookie"}),
        )
        is True
    )
