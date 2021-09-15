from datasets_preview_backend.utils import get_int_value, get_str_value, get_token

# from starlette.requests import  Request


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


# def test_get_token():
# TODO: how to test this? I don't know how to create or mock a request
# TestClient requires an app as the input (https://www.starlette.io/testclient/)
# r = Request({"type": "http", "headers": {"Authorization": "Bearer some_token"}})
# assert get_token(r) == "some_token"
# assert get_token({"headers": {"Authorization": "Basic some_token"}}) is None
# assert get_token({"headers": {"Authorization": "Bearersome_token"}}) is None
# assert get_token({"headers": {}}) is None
