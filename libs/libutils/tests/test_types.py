from typing import get_args

from libutils.types import (
    ClassLabelColumnType,
    ColumnDict,
    CommonColumnType,
    TimestampColumnType,
    TimestampUnit,
)


def test_timestamp_unit() -> None:
    assert get_args(TimestampUnit) == ("s", "ms", "us", "ns")
    assert set(get_args(TimestampUnit)) == {"s", "ms", "us", "ns"}
    assert list(get_args(TimestampUnit)) == ["s", "ms", "us", "ns"]
    assert "ms" in get_args(TimestampUnit)


def test_column_type() -> None:
    assert set(get_args(CommonColumnType)) == {
        "JSON",
        "BOOL",
        "INT",
        "FLOAT",
        "STRING",
        "IMAGE_URL",
        "RELATIVE_IMAGE_URL",
        "AUDIO_RELATIVE_SOURCES",
    }
    assert set(get_args(ClassLabelColumnType)) == {"CLASS_LABEL"}
    assert set(get_args(TimestampColumnType)) == {"TIMESTAMP"}


def test_column_dict() -> None:
    # allowed
    col: ColumnDict = {"name": "mycol", "type": "JSON"}
    labels: ColumnDict = {
        "name": "mycol",
        "type": "CLASS_LABEL",
        "labels": ["positive", "negative", "neutral"],
    }
    timestamp: ColumnDict = {
        "name": "mycol",
        "type": "TIMESTAMP",
        "tz": None,
        "unit": "ms",
    }
    # not allowed
    missing_field: ColumnDict = {
        "name": "mycol",
        "type": "TIMESTAMP",
        "tz": None,
    }  # type: ignore
    wrong_type: ColumnDict = {
        "name": "mycol",
        "type": "JSON",  # type: ignore
        "tz": None,
        "unit": "ms",
    }

    # nothing to test, we just want to ensure that mypy doesn't break
    assert col
    assert labels
    assert timestamp
    assert missing_field
    assert wrong_type
