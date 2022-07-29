import re
from typing import Any, List, Optional, get_args

import pandas  # type: ignore
from datasets import Value
from libutils.types import ColumnDict, TimestampColumnType, TimestampUnit

from worker.deprecated.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnInferenceError,
    ColumnTypeError,
)

# pandas types: see https://github.com/VirtusLab/pandas-stubs/issues/172


TimestampTz = Optional[str]


def cast_to_timestamp_unit(value: str) -> TimestampUnit:
    if value == "s":
        return "s"
    elif value == "ms":
        return "ms"
    elif value == "us":
        return "us"
    elif value == "ns":
        return "ns"
    raise ValueError("string cannot be cast to timestamp unit")


def get_tz(ts: pandas.Timestamp) -> TimestampTz:
    return None if ts.tz is None else str(ts.tz)


def infer_from_values(
    values: List[Any],
) -> TimestampTz:
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")
    if any(not isinstance(value, pandas.Timestamp) for value in values):
        raise ColumnInferenceError("some values are not timestamps, cannot infer column type")
    timezones = {value.tz for value in values if isinstance(value, pandas.Timestamp) and value.tz is not None}
    if len(timezones) > 1:
        raise ColumnInferenceError("several timezones found in the values, cannot infer column type")
    elif len(timezones) == 1:
        return str(list(timezones)[0].tzinfo)
    return None


class TimestampColumn(Column):
    type: TimestampColumnType
    unit: TimestampUnit
    tz: TimestampTz

    def __init__(self, name: str, feature: Any, values: List[Any]):
        if not feature:
            tz = infer_from_values(values)
            unit = "s"
        if not isinstance(feature, Value):
            raise ColumnTypeError("feature type mismatch")
        # see https://github.com/huggingface/datasets/blob/master/src/datasets/features/features.py
        timestamp_matches = re.search(r"^timestamp\[(.*)\]$", feature.dtype)
        if not timestamp_matches:
            raise ColumnTypeError("feature type mismatch")
        timestamp_internals = timestamp_matches[1]
        timestampUnits = get_args(TimestampUnit)
        internals_matches = re.search(r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals)
        if timestamp_internals in timestampUnits:
            unit = timestamp_internals
            tz = None
        elif internals_matches:
            unit = internals_matches[1]
            tz = internals_matches[2]
        else:
            raise ColumnTypeError("feature type mismatch")

        self.name = name
        self.unit = cast_to_timestamp_unit(unit)
        self.tz = tz
        self.type = "TIMESTAMP"

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        if value is None:
            return None
        if not isinstance(value, pandas.Timestamp):
            raise CellTypeError("value must be a pandas Timestamp object")
        posix_timestamp_in_seconds = value.timestamp()
        if self.unit == "s":
            return posix_timestamp_in_seconds
        elif self.unit == "ms":
            return posix_timestamp_in_seconds * 1_000
        elif self.unit == "us":
            return posix_timestamp_in_seconds * 1_000_000
        elif self.unit == "ns":
            return posix_timestamp_in_seconds * 1_000_000_000

    def as_dict(self) -> ColumnDict:
        return {"name": self.name, "type": self.type, "tz": self.tz, "unit": self.unit}
