from typing import Any

from datasets_preview_backend.io.asset import create_asset_file
from datasets_preview_backend.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnType,
    ColumnTypeError,
    check_feature_type,
)

COLUMN_NAMES = ["image"]


class ImageBytesColumn(Column):
    def __init__(self, name: str, feature: Any):
        if name not in COLUMN_NAMES:
            raise ColumnTypeError("feature name mismatch")
        try:
            check_feature_type(feature["filename"], "Value", ["string"])
            check_feature_type(feature["data"], "Value", ["binary"])
        except Exception:
            raise ColumnTypeError("feature type mismatch")
        self.name = name
        self.type = ColumnType.RELATIVE_IMAGE_URL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        try:
            filename = value["filename"]
            data = value["data"]
        except Exception:
            raise CellTypeError("image cell must contain 'filename' and 'data' fields")
        if type(filename) != str:
            raise CellTypeError("'filename' field must be a string")
        if type(data) != bytes:
            raise CellTypeError("'data' field must be a bytes")

        # this function can raise, we don't catch it
        return create_asset_file(dataset_name, config_name, split_name, row_idx, self.name, filename, data)
