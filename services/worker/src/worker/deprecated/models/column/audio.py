from typing import Any, List

from datasets import Audio
from numpy import ndarray  # type:ignore

from worker.deprecated.models.asset import create_audio_files
from worker.deprecated.models.column.default import (
    Cell,
    CellTypeError,
    ColumnInferenceError,
    ColumnTypeError,
    CommonColumn,
)


def check_value(value: Any) -> None:
    if value is None:
        return
    try:
        path = value["path"]
        array = value["array"]
        sampling_rate = value["sampling_rate"]
    except Exception as e:
        raise CellTypeError("audio cell must contain 'path' and 'array' fields") from e
    if path is not None and type(path) != str:
        raise CellTypeError("'path' field must be a string or None")
    if type(array) != ndarray:
        raise CellTypeError("'array' field must be a numpy.ndarray")
    if type(sampling_rate) != int:
        raise CellTypeError("'sampling_rate' field must be an integer")


def infer_from_values(values: List[Any]) -> None:
    for value in values:
        check_value(value)
    if values and all(value is None for value in values):
        raise ColumnInferenceError("all the values are None, cannot infer column type")


class AudioColumn(CommonColumn):
    def __init__(self, name: str, feature: Any, values: List[Any]):
        if feature:
            if not isinstance(feature, Audio):
                raise ColumnTypeError("feature type mismatch")
        else:
            infer_from_values(values)
        self.name = name
        self.type = "AUDIO_RELATIVE_SOURCES"

    def get_cell_value(self, dataset_name: str, config_name: str, split_name: str, row_idx: int, value: Any) -> Cell:
        if value is None:
            return None
        check_value(value)
        array = value["array"]
        sampling_rate = value["sampling_rate"]
        # this function can raise, we don't catch it
        return create_audio_files(
            dataset_name, config_name, split_name, row_idx, self.name, array, sampling_rate, "assets"
        )
