from typing import Any, List

from datasets_preview_backend.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
    check_feature_type,
)


def check_value(value: Any) -> None:
    if value is not None and type(value) != str:
        raise CellTypeError("value type mismatch")


def check_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class StringColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            try:
                check_feature_type(feature, "Value", ["string", "large_string"])
            except Exception:
                raise ColumnTypeError("feature type mismatch")
        # else: we can infer from values
        check_values(values)
        self.name = name
        self.type = ColumnType.STRING

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        return value
