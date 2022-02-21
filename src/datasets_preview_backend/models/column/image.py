from typing import Any, List

from PIL import Image  # type: ignore

from datasets_preview_backend.io.asset import create_image_file
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
    if value is None:
        return
    if not isinstance(value, Image.Image):
        raise CellTypeError("image cell must be a PIL image")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class ImageColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            try:
                check_feature_type(feature, "Image", [])
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
        # this function can raise, we don't catch it
        return create_image_file(dataset_name, config_name, split_name, row_idx, self.name, "image.jpg", value)
