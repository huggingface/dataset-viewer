from typing import Any, Dict, List, TypedDict, Union


class DatasetItem(TypedDict):
    dataset: str


class InfoItem(TypedDict):
    dataset: str
    info: Dict[str, Any]


# Content of endpoint responses
class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


InfoContent = InfoItem


class ConfigsContent(TypedDict):
    dataset: str
    configs: List[str]


class SplitsContent(TypedDict):
    dataset: str
    config: str
    splits: List[str]


class RowsContent(TypedDict):
    dataset: str
    config: str
    split: str
    rows: List[Any]


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
