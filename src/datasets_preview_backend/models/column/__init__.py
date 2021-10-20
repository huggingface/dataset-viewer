from typing import Any, List, Union

from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.models.column.bool import BoolColumn
from datasets_preview_backend.models.column.class_label import ClassLabelColumn
from datasets_preview_backend.models.column.default import (
    Cell,
    Column,
    ColumnType,
    ColumnTypeError,
    JsonColumn,
)
from datasets_preview_backend.models.column.float import FloatColumn
from datasets_preview_backend.models.column.image_array2d import ImageArray2DColumn
from datasets_preview_backend.models.column.image_array3d import ImageArray3DColumn
from datasets_preview_backend.models.column.image_bytes import ImageBytesColumn
from datasets_preview_backend.models.column.image_url import ImageUrlColumn
from datasets_preview_backend.models.column.int import IntColumn
from datasets_preview_backend.models.column.string import StringColumn
from datasets_preview_backend.models.info import Info

column_classes = [
    ClassLabelColumn,
    ImageBytesColumn,
    ImageArray2DColumn,
    ImageArray3DColumn,
    ImageUrlColumn,
    BoolColumn,
    IntColumn,
    FloatColumn,
    StringColumn,
]


def get_column(name: str, feature: Any) -> Column:
    # try until one works
    for column_class in column_classes:
        try:
            return column_class(name, feature)
        except ColumnTypeError:
            pass
    # none has worked
    return Column(name, feature)


def get_columns_from_info(info: Info) -> Union[List[Column], None]:
    try:
        if info["features"] is None:
            return None
        return [get_column(name, feature) for (name, feature) in info["features"].items()]
    except Exception as err:
        # note that no exception will be raised if features exists but is empty
        raise Status400Error("features could not be inferred from dataset config info", err)


# explicit re-export
__all__ = ["Column", "Cell", "ColumnType", "JsonColumn"]
