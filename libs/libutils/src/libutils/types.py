from typing import Any, Dict, List, Optional, TypedDict, Union
from libutils.enums import TimestampUnit


class CommonColumnDict(TypedDict):
    name: str
    type: str


class ClassLabelColumnDict(CommonColumnDict):
    labels: List[str]


class TimestampColumnDict(CommonColumnDict):
    unit: TimestampUnit
    tz: Optional[str]


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
