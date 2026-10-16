# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

"""Mock the httpx2 clients, since pytest-httpx only supports httpx (not httpx2).

See the `httpx_mock` fixture (defined in tests/conftest.py): it yields an `HTTPXMock` instance on which a test
registers the responses to return to the requests made by the code under test.

```python
async def test_foo(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url="https://auth.check", status_code=404, text="{}")
    # the requests to https://auth.check made by the code under test now get a 404 response
```
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import httpx2 as httpx
from pytest import MonkeyPatch

Callback = Callable[[httpx.Request], httpx.Response]
URLTypes = Union[str, httpx.URL]

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def request_callback(request: httpx.Request) -> httpx.Response:
    # return 404 if a token has been provided,
    # and 200 if none has been provided
    # there is no logic behind this behavior, it's just to test if the
    # tokens are correctly passed to the auth_check service
    body = '{"orgs": [{"name": "org1"}], "auth": {"type": "access_token", "accessToken": {"fineGrained": {"scoped": [{"entity": {"type": "org", "name": "org1"}, "permissions": ["repo.write"]}]}}}}'
    if request.headers.get("authorization"):
        return httpx.Response(status_code=404, text=body)
    return httpx.Response(status_code=200, text=body)


def _parse_url(url: Optional[URLTypes]) -> Optional[httpx.URL]:
    if url is None or isinstance(url, httpx.URL):
        return url
    return httpx.URL(url)


def _port(url: httpx.URL) -> Optional[int]:
    """the port of an URL, with the default port of the scheme if it is not specified"""
    return url.port if url.port is not None else DEFAULT_PORTS.get(url.scheme.lower())


def _url_matches(expected: Optional[httpx.URL], actual: httpx.URL) -> bool:
    """check that a requested URL is the mocked one, tolerating the differences of normalization between the
    two (an implicit trailing slash, or an explicit default port: https://host is https://host:443/)"""
    if expected is None:
        return True  # no URL was mocked: any request matches
    if expected == actual:
        return True
    return (
        expected.scheme.lower() == actual.scheme.lower()
        and expected.host.lower() == actual.host.lower()
        and _port(expected) == _port(actual)
        and expected.path.rstrip("/") == actual.path.rstrip("/")
        and expected.params == actual.params
    )


def _response_callback(**response_kwargs: Any) -> Callback:
    def callback(request: httpx.Request) -> httpx.Response:
        # a new response is built for every request, as a httpx.Response can only be consumed once
        return httpx.Response(**response_kwargs)

    return callback


@dataclass
class _MockedRequest:
    """a request matcher (method and URL) and the way to answer the requests it matches"""

    callback: Callback
    method: Optional[str] = None
    url: Optional[httpx.URL] = None
    is_reusable: bool = False
    is_optional: bool = False
    nb_matches: int = field(default=0, init=False)

    def matches(self, request: httpx.Request) -> bool:
        if self.nb_matches and not self.is_reusable:
            return False  # this response has already been requested, and cannot be reused
        if self.method is not None and request.method != self.method.upper():
            return False
        return _url_matches(self.url, request.url)

    def answer(self, request: httpx.Request) -> httpx.Response:
        self.nb_matches += 1
        return self.callback(request)

    def __str__(self) -> str:
        return (
            f"{self.method or 'ANY'} {self.url or 'ANY'}"
            f" (requested {self.nb_matches} time(s),"
            f" {'reusable' if self.is_reusable else 'single use'},"
            f" {'optional' if self.is_optional else 'required'})"
        )


class HTTPXMock:
    """Mock the httpx2 clients, so that the requests made by the code under test are answered with the
    registered responses, instead of being sent over the network.

    It is a reduced version of the pytest-httpx fixture, that cannot be used here since it does not support
    httpx2: see `add_response`, `add_callback`, `get_requests` and `assert_all_responses_were_requested`.
    """

    def __init__(self) -> None:
        self._mocked_requests: list[_MockedRequest] = []
        self._requests: list[httpx.Request] = []

    def add_response(
        self,
        status_code: int = 200,
        *,
        method: Optional[str] = None,
        url: Optional[URLTypes] = None,
        headers: Any = None,
        content: Optional[Union[bytes, str]] = None,
        text: Optional[str] = None,
        json: Any = None,
        is_reusable: bool = False,
        is_optional: bool = False,
    ) -> None:
        """register the response to return to the requests that match the method and the URL

        Args:
            status_code (`int`, *optional*): the status code of the response. Defaults to 200.
            method (`str`, *optional*): the HTTP method of the requests to answer (None: any method).
            url (`str` or `URL`, *optional*): the URL of the requests to answer (None: any URL).
            headers (`Headers`, *optional*): the headers of the response (any value accepted by httpx: the type is
              `Any`, as `httpx.HeaderTypes` cannot be used as a mypy annotation).
            content (`bytes` or `str`, *optional*): the raw body of the response.
            text (`str`, *optional*): the body of the response, as text.
            json (`Any`, *optional*): the body of the response, serialized as JSON.
            is_reusable (`bool`, *optional*): defaults to False, meaning that the response is only returned once:
              the next matching requests won't be answered by it. Set to True to answer all the matching requests.
            is_optional (`bool`, *optional*): defaults to False, meaning that the test fails if this response is
              never requested. Set to True to allow the test to pass even if it is not requested.
        """
        response_kwargs: dict[str, Any] = {"status_code": status_code}
        if headers is not None:
            response_kwargs["headers"] = headers
        if content is not None:
            response_kwargs["content"] = content
        if text is not None:
            response_kwargs["text"] = text
        if json is not None:
            response_kwargs["json"] = json
        self._mocked_requests.append(
            _MockedRequest(
                callback=_response_callback(**response_kwargs),
                method=method,
                url=_parse_url(url),
                is_reusable=is_reusable,
                is_optional=is_optional,
            )
        )

    def add_callback(
        self,
        callback: Callback,
        *,
        method: Optional[str] = None,
        url: Optional[URLTypes] = None,
        is_reusable: bool = False,
        is_optional: bool = False,
    ) -> None:
        """register a callback to build the response of the requests that match the method and the URL

        Args:
            callback (`Callback`): takes the received `httpx.Request` and returns the `httpx.Response` to answer
              with. It can raise an exception, to simulate a connection error for example.
            method (`str`, *optional*): the HTTP method of the requests to answer (None: any method).
            url (`str` or `URL`, *optional*): the URL of the requests to answer (None: any URL).
            is_reusable (`bool`, *optional*): defaults to False, meaning that the callback is only called once:
              the next matching requests won't be answered by it. Set to True to answer all the matching requests.
            is_optional (`bool`, *optional*): defaults to False, meaning that the test fails if this callback is
              never called. Set to True to allow the test to pass even if it is not called.
        """
        self._mocked_requests.append(
            _MockedRequest(
                callback=callback, method=method, url=_parse_url(url), is_reusable=is_reusable, is_optional=is_optional
            )
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """answer a request with the first matching registered response (used by the patched clients)"""
        self._requests.append(request)
        for mocked_request in self._mocked_requests:
            if mocked_request.matches(request):
                return mocked_request.answer(request)
        raise AssertionError(
            f"no response was registered for the request {request.method} {request.url} in {self}."
            " Use add_response or add_callback to register one."
        )

    def get_requests(self, *, method: Optional[str] = None, url: Optional[URLTypes] = None) -> list[httpx.Request]:
        """the requests received by the mock, in order, optionally filtered by method and/or URL

        Args:
            method (`str`, *optional*): only keep the requests with this HTTP method (None: any method).
            url (`str` or `URL`, *optional*): only keep the requests to this URL (None: any URL).
        """
        expected_url = _parse_url(url)
        return [
            request
            for request in self._requests
            if (method is None or request.method == method.upper()) and _url_matches(expected_url, request.url)
        ]

    def get_request(self, *, method: Optional[str] = None, url: Optional[URLTypes] = None) -> httpx.Request:
        """the unique request received by the mock for this method and/or URL

        Raises:
            ValueError: if there is not exactly one matching request.
        """
        requests = self.get_requests(method=method, url=url)
        if len(requests) != 1:
            raise ValueError(f"expected exactly one request, got {len(requests)}")
        return requests[0]

    def clear(self) -> None:
        """forget the registered responses, and the requests received so far"""
        self._mocked_requests.clear()
        self._requests.clear()

    def assert_all_responses_were_requested(self) -> None:
        """fail if a registered, non-optional, response was never requested

        It's the default behavior of pytest-httpx: it helps to detect the tests that do not exercise the code they
        think they exercise (see `is_optional` to opt out for a given response).
        """
        unused = [m for m in self._mocked_requests if not m.is_optional and not m.nb_matches]
        if unused:
            details = ", ".join(str(m) for m in unused)
            raise AssertionError(f"the following registered response(s) were never requested: {details}.")

    @contextmanager
    def patched(self, monkeypatch: MonkeyPatch) -> Iterator["HTTPXMock"]:
        """patch the httpx2 clients during the context, so that they are answered by this mock

        Only the transport used by the clients is replaced, the real httpx clients are kept: the requests are still
        built, serialized, timed out, authenticated and redirected as usual, and only the network layer is mocked.
        It works whatever the way the client is created by the code under test.
        """
        transport = httpx.MockTransport(self.handle_request)

        def mocked_transport_for_url(client: Any, url: httpx.URL) -> httpx.MockTransport:
            return transport

        monkeypatch.setattr(httpx.Client, "_transport_for_url", mocked_transport_for_url)
        monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", mocked_transport_for_url)
        try:
            yield self
        finally:
            self.assert_all_responses_were_requested()

    def __str__(self) -> str:
        if not self._mocked_requests:
            return "HTTPXMock(no registered response)"
        registered = ", ".join(str(m) for m in self._mocked_requests)
        return f"HTTPXMock({registered})"
