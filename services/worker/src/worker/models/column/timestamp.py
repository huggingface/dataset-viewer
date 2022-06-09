import pandas
import re
from typing import Any, List, Tuple, Union

from datasets import Value
from worker.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnDict,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
)

TimestampUnit = str
TimestampTz = Union[str, None]
TimestampUnitTz = Tuple[TimestampUnit, TimestampTz]


def get_unit(ts: pandas.Timestamp) -> TimestampUnit:
    isoformat = ts.resolution.isoformat()
    if isoformat == "P0DT0H0M0.000000001S":
        return "ns"
    elif isoformat == "P0DT0H0M0.000001S":
        return "us"
    elif isoformat == "P0DT0H0M0.001S":
        return "ms"
    elif isoformat == "P0DT0H0M1S":
        return "s"
    raise CellTypeError("timestamp unit could not be inferred")


def get_tz(ts: pandas.Timestamp) -> TimestampTz:
    return None if ts.tz is None else str(ts.tz)


def check_value(value: Any, reference_unit_tz: Union[None, TimestampUnitTz] = None) -> Union[None, TimestampUnitTz]:
    if value is None:
        return None
    if not isinstance(value, pandas.Timestamp):
        raise CellTypeError("value must be a pandas Timestamp object")
    unit_tz = get_unit(value), get_tz(value)
    if reference_unit_tz and reference_unit_tz != unit_tz:
        raise CellTypeError("value (timestamp) must have the correct unit and tz settings")
    return unit_tz


def infer_from_values(
    values: List[Any],
) -> Tuple[TimestampUnit, TimestampTz]:
    unit_tz: Union[Tuple[TimestampUnit, TimestampTz], None] = None
    for value in values:
        new_unit_tz = check_value(value, unit_tz)
        unit_tz = new_unit_tz
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")
    if unit_tz is None:
        raise ColumnInferenceError("no value is a timestamp, cannot infer column type")
    return unit_tz


class TimestampColumn(Column):
    unit: TimestampUnit
    tz: TimestampTz

    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            if not isinstance(feature, Value):
                raise ColumnTypeError("feature type mismatch")
            # see https://github.com/huggingface/datasets/blob/master/src/datasets/features/features.py
            timestamp_matches = re.search(r"^timestamp\[(.*)\]$", feature.dtype)
            if not timestamp_matches:
                raise ColumnTypeError("feature type mismatch")
            timestamp_internals = timestamp_matches[1]
            internals_matches = re.search(r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals)
            if timestamp_internals in ["s", "ms", "us", "ns"]:
                unit = timestamp_internals
                tz = None
            elif internals_matches:
                unit = internals_matches[1]
                tz = internals_matches[2]
            else:
                raise ColumnTypeError("feature type mismatch")
        else:
            unit, tz = infer_from_values(values)
        self.name = name
        self.unit = unit
        self.tz = tz
        self.type = ColumnType.TIMESTAMP

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value, (self.unit, self.tz))
        return value

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        return value

    def as_dict(self) -> ColumnDict:
        return {"name": self.name, "tz": self.tz, "unit": self.unit}
