# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libutils.utils import (
    get_bool_value,
    get_int_value,
    get_str_list_value,
    get_str_or_none_value,
    get_str_value,
)


def test_get_bool_value() -> None:
    assert get_bool_value({"KEY": "False"}, "KEY", True) is False
    assert get_bool_value({"KEY": "True"}, "KEY", False) is True
    assert get_bool_value({"KEY": "true"}, "KEY", False) is True
    assert get_bool_value({"KEY": True}, "KEY", False) is True
    assert get_bool_value({"KEY": "True"}, "DOESNOTEXIST", False) is False
    assert get_bool_value({"KEY": ""}, "KEY", False) is False
    assert get_bool_value({}, "KEY", False) is False
    assert get_bool_value({}, "KEY", True) is True


def test_get_int_value() -> None:
    default = 456
    assert get_int_value({"KEY": "123"}, "KEY", default) == 123
    assert get_int_value({"KEY": 123}, "KEY", default) == 123
    assert get_int_value({"KEY": "123"}, "DOESNOTEXIST", default) == default
    assert get_int_value({"KEY": ""}, "KEY", default) == default
    assert get_int_value({}, "KEY", default) == default


def test_get_str_value() -> None:
    default = "string"
    assert get_str_value({}, "KEY", default) == default
    # Empty string is ignored
    assert get_str_value({"KEY": ""}, "KEY", default) == default
    assert get_str_value({"KEY": "test"}, "KEY", default) == "test"
    assert get_str_value({"KEY": "  test "}, "KEY", default) == "test"
    assert get_str_value({"KEY": "None"}, "KEY", default) == "None"
    assert get_str_value({"KEY": "test"}, "DOESNOTEXIST", default) == default


def test_get_str_list_value() -> None:
    default = ["a", "b"]
    assert get_str_list_value({}, "KEY", default) == default
    # Empty string is NOT ignored
    assert get_str_list_value({"KEY": ""}, "KEY", default) == []
    assert get_str_list_value({"KEY": "test"}, "KEY", default) == ["test"]
    assert get_str_list_value({"KEY": "None"}, "KEY", default) == ["None"]
    assert get_str_list_value({"KEY": "a,b,c"}, "KEY", default) == ["a", "b", "c"]
    assert get_str_list_value({"KEY": "a  , b,  c "}, "KEY", default) == ["a", "b", "c"]
    assert get_str_list_value({"KEY": "test"}, "DOESNOTEXIST", default) == default


def test_get_str_or_none_value() -> None:
    default = "string"
    assert get_str_or_none_value({}, "KEY", default) == default
    # Empty string is ignored
    assert get_str_or_none_value({"KEY": ""}, "KEY", default) == default
    assert get_str_or_none_value({"KEY": "test"}, "KEY", default) == "test"
    assert get_str_or_none_value({"KEY": "None"}, "KEY", default) == "None"
    assert get_str_or_none_value({"KEY": "test"}, "DOESNOTEXIST", default) == default
    assert get_str_or_none_value({}, "KEY", None) is None
