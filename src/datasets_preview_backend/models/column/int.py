from typing import Any

from datasets_preview_backend.models.column.default import (
    Cell,
    Column,
    ColumnType,
    ColumnTypeError,
    check_feature_type,
)


class IntColumn(Column):
    def __init__(self, name: str, feature: Any):
        try:
            check_feature_type(
                feature,
                "Value",
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
            )
        except Exception:
            raise ColumnTypeError("feature type mismatch")
        self.name = name
        self.type = ColumnType.INT

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        return int(value)
