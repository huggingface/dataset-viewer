from typing import Any, List

import numpy  # type: ignore
from datasets import Array3D
from PIL import Image  # type: ignore

from worker.models.asset import create_image_file
from worker.models.column.default import (
    Cell,
    CellTypeError,
    ColumnInferenceError,
    ColumnTypeError,
    CommonColumn,
    check_dtype,
)

COLUMN_NAMES = ["img"]


def check_value(value: Any) -> None:
    if value is not None and (
        not isinstance(value, list)
        or len(value) == 0
        or not isinstance(value[0], list)
        or len(value[0]) == 0
        or not isinstance(value[0][0], list)
        or len(value[0][0]) == 0
        or type(value[0][0][0]) != int
    ):
        raise CellTypeError("value must contain 3D array of integers")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class ImageArray3DColumn(CommonColumn):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if name not in COLUMN_NAMES:
            raise ColumnTypeError("feature name mismatch")
        if feature:
            if not check_dtype(feature, ["uint8"], Array3D):
                raise ColumnTypeError("feature type mismatch")
        else:
            infer_from_values(values)
        # we also have shape in the feature: shape: [32, 32, 3] for cifar10
        self.name = name
        self.type = "RELATIVE_IMAGE_URL"

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        if value is None:
            return None
        array = numpy.asarray(value, dtype=numpy.uint8)
        mode = "RGB"
        image = Image.fromarray(array, mode)
        filename = "image.jpg"

        return create_image_file(dataset_name, config_name, split_name, row_idx, self.name, filename, image)
