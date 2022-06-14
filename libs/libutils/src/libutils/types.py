from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

TimestampUnit = Literal["s", "ms", "us", "ns"]
CommonColumnType = Literal[
    "JSON", "BOOL", "INT", "FLOAT", "STRING", "IMAGE_URL", "RELATIVE_IMAGE_URL", "AUDIO_RELATIVE_SOURCES"
]
ClassLabelColumnType = Literal["CLASS_LABEL"]
TimestampColumnType = Literal["TIMESTAMP"]


class _BaseColumnDict(TypedDict):
    name: str


class CommonColumnDict(_BaseColumnDict):
    type: CommonColumnType


class ClassLabelColumnDict(_BaseColumnDict):
    type: ClassLabelColumnType
    labels: List[str]


class TimestampColumnDict(_BaseColumnDict):
    type: TimestampColumnType
    unit: TimestampUnit
    tz: Optional[str]


ColumnType = Union[CommonColumnType, ClassLabelColumnType, TimestampColumnType]
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
