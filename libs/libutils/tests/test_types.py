from libutils.types import ColumnDict
from libutils.enums import TimestampUnit


def test_column_dict() -> None:
    # allowed
    col: ColumnDict = {"name": "mycol", "type": "some type that should be in an enum"}
    labels: ColumnDict = {
        "name": "mycol",
        "type": "some other type that should be in an enum",
        "labels": ["positive", "negative", "neutral"],
    }
    timestamp: ColumnDict = {
        "name": "mycol",
        "type": "some other type that should be in an enum",
        "tz": None,
        "unit": TimestampUnit["ms"],
    }
    # not allowed
    broken_timestamp: ColumnDict = {
        "name": "mycol",
        "type": "some other type that should be in an enum",
        "tz": None,
    }  # type: ignore

    # nothing to test, we just want to ensure that mypy doesn't break
    assert col
    assert labels
    assert timestamp
    assert broken_timestamp
