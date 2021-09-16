from typing import Dict

from starlette.datastructures import Headers
from starlette.requests import Request

from datasets_preview_backend.utils import get_int_value, get_str_value, get_token


def test_get_int_value():
    default = 456
    assert get_int_value({"KEY": "123"}, "KEY", default) == 123
    assert get_int_value({"KEY": 123}, "KEY", default) == 123
    assert get_int_value({"KEY": "123"}, "DOESNOTEXIST", default) == default
    assert get_int_value({"KEY": ""}, "KEY", default) == default
    assert get_int_value({}, "KEY", default) == default
    # default value type is not constrained
    assert get_int_value({}, "KEY", None) is None
    assert get_int_value({}, "KEY", "string") == "string"


def test_get_str_value():
    default = "string"
    assert get_str_value({}, "KEY", default) == default
    # Empty string is ignored
    assert get_str_value({"KEY": ""}, "KEY", default) == default
    assert get_str_value({"KEY": "test"}, "KEY", default) == "test"
    assert get_str_value({"KEY": "None"}, "KEY", default) == "None"
    assert get_str_value({"KEY": "test"}, "DOESNOTEXIST", default) == default
    # default value type is not constrained
    assert get_str_value({}, "KEY", None) is None
    assert get_str_value({}, "KEY", 123) == 123


def build_request(headers: Dict = None) -> Request:
    if headers is None:
        headers = {}
    return Request({"type": "http", "headers": Headers(headers).raw})


def test_get_token():
    assert get_token(build_request({"Authorization": "Bearer some_token"})) == "some_token"
    assert get_token(build_request({"Authorization": "beArER some_token"})) == "some_token"
    assert get_token(build_request({"Authorization": "Basic some_token"})) is None
    assert get_token(build_request({"Authorization": "Bearersome_token"})) is None
    assert get_token(build_request({})) is None
