# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable

import pytest
from starlette.requests import Request

from libapi.exceptions import InvalidParameterError
from libapi.request import get_request_parameter_length, get_request_parameter_offset


@pytest.fixture
def build_request() -> Callable[..., Request]:
    def _build_request(query_string: str = "") -> Request:
        scope = {"type": "http"}
        if query_string:
            scope["query_string"] = query_string  # .encode("utf-8")
        return Request(scope)

    return _build_request


@pytest.mark.parametrize("length, expected_value", [("50", 50)])
def test_get_request_parameter_length(length: str, expected_value: int, build_request: Callable[..., Request]) -> None:
    request = build_request(query_string=f"length={length}")
    assert get_request_parameter_length(request) == expected_value


@pytest.mark.parametrize(
    "length, expected_error_message",
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


@pytest.mark.parametrize("offset, expected_value", [("50", 50)])
def test_get_request_parameter_offset(offset: str, expected_value: int, build_request: Callable[..., Request]) -> None:
    request = build_request(query_string=f"offset={offset}")
    assert get_request_parameter_offset(request) == expected_value


@pytest.mark.parametrize(
    "offset, expected_error_message",
    [("abc", "Parameter 'offset' must be integer"), ("-1", "Parameter 'offset' must be positive")],
)
def test_get_request_parameter_offset_raises(
    offset: str, expected_error_message: str, build_request: Callable[..., Request]
) -> None:
    request = build_request(query_string=f"offset={offset}")
    with pytest.raises(InvalidParameterError, match=expected_error_message):
        _ = get_request_parameter_offset(request)
