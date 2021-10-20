from typing import Any

from datasets_preview_backend.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnType,
    ColumnTypeError,
    check_feature_type,
)

COLUMN_NAMES = ["image_url"]


class ImageUrlColumn(Column):
    def __init__(self, name: str, feature: Any):
        if name not in COLUMN_NAMES:
            raise ColumnTypeError("feature name mismatch")
        try:
            check_feature_type(feature, "Value", ["string"])
        except Exception:
            raise ColumnTypeError("feature type mismatch")
        self.name = name
        self.type = ColumnType.IMAGE_URL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        # TODO: also manage nested dicts and other names
        if type(value) != str:
            raise CellTypeError("image URL column must be a string")
        return value
