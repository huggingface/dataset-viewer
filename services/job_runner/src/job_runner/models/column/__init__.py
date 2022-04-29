from typing import List, Union

from datasets import DatasetInfo, Features
from libutils.exceptions import Status400Error

from job_runner.config import ROWS_MAX_NUMBER
from job_runner.models.column.audio import AudioColumn
from job_runner.models.column.bool import BoolColumn
from job_runner.models.column.class_label import ClassLabelColumn
from job_runner.models.column.default import (
    Cell,
    CellTypeError,
    Column,
    ColumnDict,
    ColumnInferenceError,
    ColumnType,
    ColumnTypeError,
)
from job_runner.models.column.float import FloatColumn
from job_runner.models.column.image import ImageColumn
from job_runner.models.column.image_array2d import ImageArray2DColumn
from job_runner.models.column.image_array3d import ImageArray3DColumn
from job_runner.models.column.image_url import ImageUrlColumn
from job_runner.models.column.int import IntColumn
from job_runner.models.column.string import StringColumn
from job_runner.models.row import Row

column_classes = [
    AudioColumn,
    ClassLabelColumn,
    ImageColumn,
    ImageArray2DColumn,
    ImageArray3DColumn,
    ImageUrlColumn,
    BoolColumn,
    IntColumn,
    FloatColumn,
    StringColumn,
]

FeaturesOrNone = Union[Features, None]

MAX_ROWS_FOR_TYPE_INFERENCE_AND_CHECK = ROWS_MAX_NUMBER


def get_column(column_name: str, features: FeaturesOrNone, rows: List[Row]) -> Column:
    feature = None if features is None else features[column_name]
    try:
        values = [row[column_name] for row in rows[:MAX_ROWS_FOR_TYPE_INFERENCE_AND_CHECK]]
    except KeyError as e:
        raise Status400Error("one column is missing in the dataset rows", e) from e

    # try until one works
    for column_class in column_classes:
        try:
            return column_class(column_name, feature, values)
        except (ColumnTypeError, CellTypeError, ColumnInferenceError):
            pass
    # none has worked
    return Column(column_name, feature, values)


def get_columns(info: DatasetInfo, rows: List[Row]) -> List[Column]:
    if info.features is None:
        if not rows:
            return []
        else:
            column_names = list(rows[0].keys())
    else:
        column_names = list(info.features.keys())
        # check, just in case
        if rows and info.features.keys() != rows[0].keys():
            raise Status400Error("columns from features and first row don't match")
    return [get_column(column_name, info.features, rows) for column_name in column_names]


# explicit re-export
__all__ = ["Column", "Cell", "ColumnType", "ColumnDict", "ClassLabelColumn"]
