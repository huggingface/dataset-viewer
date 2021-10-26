from enum import Enum, auto
from typing import Any, List, TypedDict


class ColumnType(Enum):
    JSON = auto()  # default
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IMAGE_URL = auto()
    RELATIVE_IMAGE_URL = auto()
    CLASS_LABEL = auto()


# TODO: a set of possible cell types (problem: JSON is Any)
Cell = Any


class _BaseColumnDict(TypedDict):
    name: str
    type: str


class ColumnDict(_BaseColumnDict, total=False):
    # https://www.python.org/dev/peps/pep-0655/#motivation
    labels: List[str]


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


def check_feature_type(value: Any, type: str, dtypes: List[str]) -> None:
    if "_type" not in value or value["_type"] != type:
        raise TypeError("_type is not the expected value")
    if dtypes and ("dtype" not in value or value["dtype"] not in dtypes):
        raise TypeError("dtype is not the expected value")
