from typing import Any, Dict, List, Union

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.models.column.audio import AudioColumn
from datasets_preview_backend.models.column.bool import BoolColumn
from datasets_preview_backend.models.column.class_label import ClassLabelColumn
from datasets_preview_backend.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnDict,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
)
from datasets_preview_backend.models.column.float import FloatColumn
from datasets_preview_backend.models.column.image import ImageColumn
from datasets_preview_backend.models.column.image_array2d import ImageArray2DColumn
from datasets_preview_backend.models.column.image_array3d import ImageArray3DColumn
from datasets_preview_backend.models.column.image_bytes import ImageBytesColumn
from datasets_preview_backend.models.column.image_url import ImageUrlColumn
from datasets_preview_backend.models.column.int import IntColumn
from datasets_preview_backend.models.column.string import StringColumn
from datasets_preview_backend.models.info import Info
from datasets_preview_backend.models.row import Row

column_classes = [
    AudioColumn,
    ClassLabelColumn,
    ImageColumn,
    ImageBytesColumn,
    ImageArray2DColumn,
    ImageArray3DColumn,
    ImageUrlColumn,
    BoolColumn,
    IntColumn,
    FloatColumn,
    StringColumn,
]

Features = Dict[str, Any]
FeaturesOrNone = Union[Features, None]

MAX_ROWS_FOR_TYPE_INFERENCE_AND_CHECK = EXTRACT_ROWS_LIMIT


def get_features(info: Info) -> FeaturesOrNone:
    try:
        return None if info["features"] is None else {name: feature for (name, feature) in info["features"].items()}
    except Exception as err:
        raise Status400Error("features could not be extracted from dataset config info", err)


def get_column(column_name: str, features: FeaturesOrNone, rows: List[Row]) -> Column:
    feature = None if features is None else features[column_name]
    values = [row[column_name] for row in rows[:MAX_ROWS_FOR_TYPE_INFERENCE_AND_CHECK] if column_name in row]

    # try until one works
    for column_class in column_classes:
        try:
            return column_class(column_name, feature, values)
        except (ColumnTypeError, CellTypeError, ColumnInferenceError):
            pass
    # none has worked
    return Column(column_name, feature, values)


def get_columns(info: Info, rows: List[Row]) -> List[Column]:
    features = get_features(info)

    # order
    if features is None:
        if not rows:
            return []
        else:
            column_names = list(
                {column_name for row in rows[:MAX_ROWS_FOR_TYPE_INFERENCE_AND_CHECK] for column_name in row.keys()}
            )
    else:
        column_names = list(features.keys())
    return [get_column(column_name, features, rows) for column_name in column_names]


# explicit re-export
__all__ = ["Column", "Cell", "ColumnType", "ColumnDict", "ClassLabelColumn"]
