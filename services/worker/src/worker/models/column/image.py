from typing import Any, List

from datasets import Image
from PIL import Image as PILImage  # type: ignore

from worker.models.asset import create_image_file
from worker.models.column.default import (
    Cell,
    CellTypeError,
    ColumnInferenceError,
    ColumnTypeError,
    CommonColumn,
)


def check_value(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, PILImage.Image):
        raise CellTypeError("image cell must be a PIL image")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class ImageColumn(CommonColumn):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            if not isinstance(feature, Image):
                raise ColumnTypeError("feature type mismatch")
        else:
            infer_from_values(values)
        self.name = name
        self.type = "RELATIVE_IMAGE_URL"

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        if value is None:
            return None
        check_value(value)
        # attempt to generate one of the supported formats; if unsuccessful, throw an error
        for ext in [".jpg", ".png"]:
            try:
                image_file = create_image_file(
                    dataset_name, config_name, split_name, row_idx, self.name, "image" + ext, value
                )
            except Exception:
                continue
            else:
                break
        else:
            raise ValueError("Image cannot be written as JPEG or PNG")
        return image_file
