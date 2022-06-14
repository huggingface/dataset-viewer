from libutils.enums import (
    CommonColumnType,
    LabelsColumnType,
    TimestampColumnType,
    TimestampUnit,
)
from libutils.types import ColumnDict


def test_column_dict() -> None:
    # allowed
    col: ColumnDict = {"name": "mycol", "type": CommonColumnType.JSON.name}
    labels: ColumnDict = {
        "name": "mycol",
        "type": LabelsColumnType.CLASS_LABEL.name,
        "labels": ["positive", "negative", "neutral"],
    }
    timestamp: ColumnDict = {
        "name": "mycol",
        "type": TimestampColumnType.TIMESTAMP.name,
        "tz": None,
        "unit": TimestampUnit["ms"],
    }
    # not allowed
    missing_field: ColumnDict = {
        "name": "mycol",
        "type": TimestampColumnType.TIMESTAMP.name,
        "tz": None,
    }  # type: ignore
    wrong_type: ColumnDict = {
        "name": "mycol",
        "type": CommonColumnType.JSON.name,  # type: ignore
        "tz": None,
        "unit": TimestampUnit["ms"],
    }

    # nothing to test, we just want to ensure that mypy doesn't break
    assert col
    assert labels
    assert timestamp
    assert missing_field
    assert wrong_type
