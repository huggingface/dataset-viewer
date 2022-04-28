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
    if value is not None and type(value) != bool:
        raise CellTypeError("value type mismatch")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class BoolColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            if not check_dtype(feature, ["bool"]):
                raise ColumnTypeError("feature type mismatch")
        else:
            infer_from_values(values)
        self.name = name
        self.type = ColumnType.BOOL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        return value
