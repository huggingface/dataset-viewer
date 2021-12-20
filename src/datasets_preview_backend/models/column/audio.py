from typing import Any, List

from numpy import ndarray  # type:ignore

from datasets_preview_backend.io.asset import create_audio_files
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
    if value is not None:
        try:
            path = value["path"]
            array = value["array"]
        except Exception:
            raise CellTypeError("audio cell must contain 'path' and 'array' fields")
        if type(path) != str:
            raise CellTypeError("'path' field must be a string")
        if type(array) != ndarray:
            raise CellTypeError("'array' field must be a numpy.ndarray")


def check_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class AudioColumn(Column):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            try:
                check_feature_type(feature, "Audio", [])
            except Exception:
                raise ColumnTypeError("feature type mismatch")
        # else: we can infer from values
        check_values(values)
        self.name = name
        self.type = ColumnType.AUDIO_RELATIVE_SOURCES

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        check_value(value)
        array = value["array"]
        sampling_rate = value["sampling_rate"]
        # this function can raise, we don't catch it
        return create_audio_files(dataset_name, config_name, split_name, row_idx, self.name, array, sampling_rate)
