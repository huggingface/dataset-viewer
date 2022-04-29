from typing import Any, List

from job_runner.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
    check_dtype,
)


def check_value(value: Any) -> None:
    if value is not None and type(value) != int:
        raise CellTypeError("value type mismatch")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class IntColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            if not check_dtype(
                feature,
                [
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                ],
            ):
                raise ColumnTypeError("feature type mismatch")
        else:
            infer_from_values(values)
        self.name = name
        self.type = ColumnType.INT

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        return value
