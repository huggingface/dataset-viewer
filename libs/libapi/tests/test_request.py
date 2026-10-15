# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable, Mapping
from contextlib import nullcontext as does_not_raise
from typing import Optional

import pytest
from starlette.datastructures import Headers
from starlette.requests import Request

from libapi.exceptions import InvalidParameterError, MissingRequiredParameterError
from libapi.request import (
    get_request_ip,
    get_request_parameter,
    get_request_parameter_length,
    get_request_parameter_offset,
    get_request_source,
)


@pytest.fixture
def build_request() -> Callable[..., Request]:
    def _build_request(
        query_string: str = "",
        headers: Optional[Mapping[str, str]] = None,
        client: Optional[tuple[str, int]] = None,
    ) -> Request:
        scope = {
            "type": "http",
            "query_string": query_string,
            "headers": Headers(headers or {}).raw,
            "client": client,
        }
        return Request(scope)

    return _build_request


@pytest.mark.parametrize("expected_value", ["org-name/dataset-name"])
@pytest.mark.parametrize("required", [False, True])
def test_get_request_parameter(expected_value: str, required: bool, build_request: Callable[..., Request]) -> None:
    parameter_name = "dataset"
    request = build_request(query_string=f"{parameter_name}={expected_value}")
    assert get_request_parameter(request, parameter_name, required=required) == expected_value


@pytest.mark.parametrize("parameter", [None, "", " "])
@pytest.mark.parametrize("required", [False, True])
def test_get_request_parameter_raises(
    parameter: Optional[str], required: bool, build_request: Callable[..., Request]
) -> None:
    parameter_name = "dataset"
    expected_error_message = f"Parameter '{parameter_name}' is required"
    request = build_request(query_string=f"{parameter_name}={parameter}") if parameter is not None else build_request()
    with pytest.raises(MissingRequiredParameterError, match=expected_error_message) if required else does_not_raise():
        _ = get_request_parameter(request, parameter_name, required=required)


@pytest.mark.parametrize("length,expected_value", [("50", 50)])
def test_get_request_parameter_length(length: str, expected_value: int, build_request: Callable[..., Request]) -> None:
    request = build_request(query_string=f"length={length}")
    assert get_request_parameter_length(request) == expected_value


@pytest.mark.parametrize(
    "length,expected_error_message",
    [
        ("abc", "Parameter 'length' must be integer"),
        ("-1", "Parameter 'length' must be positive"),
        ("200", "Parameter 'length' must not be greater than 100"),
    ],
)
def test_get_request_parameter_length_raises(
    length: str, expected_error_message: str, build_request: Callable[..., Request]
) -> None:
    request = build_request(query_string=f"length={length}")
    with pytest.raises(InvalidParameterError, match=expected_error_message):
        _ = get_request_parameter_length(request)


@pytest.mark.parametrize("offset,expected_value", [("50", 50)])
def test_get_request_parameter_offset(offset: str, expected_value: int, build_request: Callable[..., Request]) -> None:
    request = build_request(query_string=f"offset={offset}")
    assert get_request_parameter_offset(request) == expected_value


@pytest.mark.parametrize(
    "offset,expected_error_message",
    [("abc", "Parameter 'offset' must be integer"), ("-1", "Parameter 'offset' must be positive")],
)
def test_get_request_parameter_offset_raises(
    offset: str, expected_error_message: str, build_request: Callable[..., Request]
) -> None:
    request = build_request(query_string=f"offset={offset}")
    with pytest.raises(InvalidParameterError, match=expected_error_message):
        _ = get_request_parameter_offset(request)


HUB_ORIGIN = "https://huggingface.co"
HUB_NETWORK = "10.0.0.0/16"
HUB_IP = "10.0.5.42"
EXTERNAL_IP = "203.0.113.7"


def test_get_request_ip(build_request: Callable[..., Request]) -> None:
    assert get_request_ip(build_request(client=("192.0.2.1", 555))) == "192.0.2.1"
    assert get_request_ip(build_request()) is None
    # the forwarded headers are not used: they are set by the callers, when the peer address comes from the connection
    request = build_request(headers={"x-forwarded-for": HUB_IP}, client=("192.0.2.1", 555))
    assert get_request_ip(request) == "192.0.2.1"


@pytest.mark.parametrize(
    "headers,client,hub_origins,hub_networks,expected_source",
    [
        # the Hub UI, in the browser: the call is cross-origin, so the Origin is always sent
        ({"origin": HUB_ORIGIN}, (EXTERNAL_IP, 555), [HUB_ORIGIN], [], "hub-browser"),
        # an external user, from a browser or from a script
        ({}, (EXTERNAL_IP, 555), [HUB_ORIGIN], [HUB_NETWORK], "external"),
        ({"origin": "https://user.example.com"}, (EXTERNAL_IP, 555), [HUB_ORIGIN], [HUB_NETWORK], "external"),
        ({"origin": HUB_ORIGIN}, (EXTERNAL_IP, 555), [], [HUB_NETWORK], "external"),  # origin not configured
        # the Hub servers, connecting from one of the known Hub networks
        ({}, (HUB_IP, 54321), [], [HUB_NETWORK], "hub-server"),
        ({}, (HUB_IP, 54321), [], [HUB_IP], "hub-server"),  # a bare address is a one-host network
        ({}, ("::1", 54321), [], ["::1/128"], "hub-server"),
        ({}, ("10.1.0.1", 54321), [], [HUB_NETWORK], "external"),  # out of the network
        ({}, (HUB_IP, 54321), [], [], "external"),  # no Hub network is configured
        ({}, (HUB_IP, 54321), [], ["not-a-network"], "external"),  # invalid configuration
        ({}, None, [], [HUB_NETWORK], "external"),  # the peer address is unknown
        # a caller cannot claim a Hub address: the headers are ignored, only the connection is trusted
        ({"x-forwarded-for": f"{HUB_IP}, {HUB_IP}"}, (EXTERNAL_IP, 555), [], [HUB_NETWORK], "external"),
        # a browser call is not counted as a server call, even from a Hub address
        ({"origin": "https://evil.example.com"}, (HUB_IP, 54321), [], [HUB_NETWORK], "external"),
        ({"origin": HUB_ORIGIN}, (HUB_IP, 54321), [HUB_ORIGIN], [HUB_NETWORK], "hub-browser"),
    ],
)
def test_get_request_source(
    headers: Mapping[str, str],
    client: Optional[tuple[str, int]],
    hub_origins: Optional[list[str]],
    hub_networks: Optional[list[str]],
    expected_source: str,
    build_request: Callable[..., Request],
) -> None:
    request = build_request(headers=headers, client=client)
    assert get_request_source(request, hub_origins=hub_origins, hub_networks=hub_networks) == expected_source


def test_get_request_source_defaults(build_request: Callable[..., Request]) -> None:
    # no configuration at all: nothing can be guessed, everything is external
    request = build_request(headers={"origin": HUB_ORIGIN}, client=(HUB_IP, 54321))
    assert get_request_source(request) == "external"
    assert get_request_source(build_request(client=(HUB_IP, 54321))) == "external"
