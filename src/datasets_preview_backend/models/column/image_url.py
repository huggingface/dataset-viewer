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

COLUMN_NAMES = ["image_url"]


def check_value(value: Any) -> None:
    if value is not None and type(value) != str:
        raise CellTypeError("image URL column must be a string")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class ImageUrlColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if name not in COLUMN_NAMES:
            raise ColumnTypeError("feature name mismatch")
        if feature:
            try:
                check_feature_type(feature, "Value", ["string"])
            except Exception:
                raise ColumnTypeError("feature type mismatch")
        else:
            # if values are strings, and the column name matches, let's say it's an image url
            infer_from_values(values)

        self.name = name
        self.type = ColumnType.IMAGE_URL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        if value is None:
            return None
        check_value(value)
        return value
