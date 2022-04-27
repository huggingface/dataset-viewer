from typing import Any, List

from datasets import ClassLabel

from libmodels.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnDict,
    ColumnType,
    ColumnTypeError,
)


def check_value(value: Any) -> None:
    if value is not None and type(value) != int:
        raise CellTypeError("class label values must be integers")


class ClassLabelColumn(Column):
    labels: List[str]

    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature is None:
            # we cannot infer from the values in that case (would be inferred as INT instead)
            raise ColumnTypeError("not a class label")
        if not isinstance(feature, ClassLabel):
            raise ColumnTypeError("feature type mismatch")
        self.labels = feature.names
        self.name = name
        self.type = ColumnType.CLASS_LABEL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        return value

    def as_dict(self) -> ColumnDict:
        return {"name": self.name, "type": self.type.name, "labels": self.labels}
