from enum import Enum, auto
from typing import Any, List

from datasets import Value
from libutils.types import ColumnDict


class ColumnType(Enum):
    JSON = auto()  # default
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IMAGE_URL = auto()
    RELATIVE_IMAGE_URL = auto()
    AUDIO_RELATIVE_SOURCES = auto()
    CLASS_LABEL = auto()
    TIMESTAMP = auto()


# TODO: a set of possible cell types (problem: JSON is Any)
Cell = Any


class Column:
    name: str
    type: ColumnType

    def __init__(self, name: str, feature: Any, values: List[Any]):
        self.name = name
        self.type = ColumnType.JSON

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        # TODO: return JSON? of pickled?
        return value

    def as_dict(self) -> ColumnDict:
        return {"name": self.name, "type": self.type.name}


# Utils


class ColumnTypeError(Exception):
    pass


class CellTypeError(Exception):
    pass


class ColumnInferenceError(Exception):
    pass


def check_dtype(feature: Any, dtypes: List[str], expected_class=None) -> bool:
    if expected_class is None:
        expected_class = Value
    return isinstance(feature, expected_class) and feature.dtype in dtypes
