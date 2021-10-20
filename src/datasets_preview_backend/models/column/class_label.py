from typing import Any, List

from datasets_preview_backend.models.column.default import (
    Cell,
    Column,
    ColumnType,
    ColumnTypeError,
    JsonColumn,
    check_feature_type,
)


class ClassLabelColumn(Column):
    labels: List[str]

    def __init__(self, name: str, feature: Any):
        try:
            check_feature_type(feature, "ClassLabel", [])
            self.labels = [str(name) for name in feature["names"]]
        except Exception:
            raise ColumnTypeError("feature type mismatch")
        self.name = name
        self.type = ColumnType.CLASS_LABEL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        return int(value)

    def to_json(self) -> JsonColumn:
        return {"name": self.name, "type": self.type.name, "labels": self.labels}
