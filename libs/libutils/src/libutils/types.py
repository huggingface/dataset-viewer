from typing import Any, Dict, List, Optional, TypedDict, Union

from libutils.enums import (
    CommonColumnType,
    LabelsColumnType,
    TimestampColumnType,
    TimestampUnit,
)


class _BaseColumnDict(TypedDict):
    name: str


class CommonColumnDict(_BaseColumnDict):
    type: CommonColumnType


class ClassLabelColumnDict(_BaseColumnDict):
    type: LabelsColumnType
    labels: List[str]


class TimestampColumnDict(_BaseColumnDict):
    type: TimestampColumnType
    unit: TimestampUnit
    tz: Optional[str]


ColumnType = Union[CommonColumnType, LabelsColumnType, TimestampColumnType]
ColumnDict = Union[CommonColumnDict, ClassLabelColumnDict, TimestampColumnDict]


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row_idx: int
    row: Dict[str, Any]
    truncated_cells: List[str]


class ColumnItem(TypedDict):
    dataset: str
    config: str
    split: str
    column_idx: int
    column: ColumnDict


class RowsResponse(TypedDict):
    columns: List[ColumnItem]
    rows: List[RowItem]


class Split(TypedDict):
    split_name: str
    rows_response: RowsResponse
    num_bytes: Optional[int]
    num_examples: Optional[int]


class SplitFullName(TypedDict):
    dataset_name: str
    config_name: str
    split_name: str
