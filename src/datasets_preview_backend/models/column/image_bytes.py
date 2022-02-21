from typing import Any, List

from datasets_preview_backend.io.asset import create_asset_file
from datasets_preview_backend.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
    check_feature_type,
)

COLUMN_NAMES = ["image"]


def check_value(value: Any) -> None:
    if value is not None:
        try:
            filename = value["filename"]
            data = value["data"]
        except Exception as e:
            raise CellTypeError("image cell must contain 'filename' and 'data' fields") from e

        if type(filename) != str:
            raise CellTypeError("'filename' field must be a string")
        if type(data) != bytes:
            raise CellTypeError("'data' field must be a bytes")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class ImageBytesColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if name not in COLUMN_NAMES:
            raise ColumnTypeError("feature name mismatch")
        if feature:
            try:
                check_feature_type(feature["filename"], "Value", ["string"])
                check_feature_type(feature["data"], "Value", ["binary"])
            except Exception as e:
                raise ColumnTypeError("feature type mismatch") from e
        else:
            infer_from_values(values)
        self.name = name
        self.type = ColumnType.RELATIVE_IMAGE_URL

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        if value is None:
            return None
        check_value(value)
        filename = value["filename"]
        data = value["data"]
        # this function can raise, we don't catch it
        return create_asset_file(dataset_name, config_name, split_name, row_idx, self.name, filename, data)
