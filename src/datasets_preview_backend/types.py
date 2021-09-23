from typing import Any, Dict, List, TypedDict, Union


class DatasetItem(TypedDict):
    dataset: str


class InfoItem(TypedDict):
    dataset: str
    info: Dict[str, Any]


class ConfigItem(TypedDict):
    dataset: str
    config: str


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str


class RowsItem(TypedDict):
    dataset: str
    config: str
    split: str
    row: Any


# Content of endpoint responses


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


InfoContent = InfoItem


class ConfigsContent(TypedDict):
    configs: List[ConfigItem]


class SplitsContent(TypedDict):
    splits: List[SplitItem]


class RowsContent(TypedDict):
    rows: List[RowsItem]


class StatusErrorContent(TypedDict):
    status_code: int
    exception: str
    message: str
    cause: str
    cause_message: str


Content = Union[
    ConfigsContent,
    DatasetsContent,
    InfoContent,
    RowsContent,
    SplitsContent,
    StatusErrorContent,
]
