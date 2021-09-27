from typing import Any, Dict, List, TypedDict, Union


class DatasetItem(TypedDict):
    dataset: str


class InfoItem(TypedDict):
    dataset: str
    config: str
    info: Dict[str, Any]


class ConfigItem(TypedDict):
    dataset: str
    config: str


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str


class RowItem(TypedDict):
    dataset: str
    config: str
    split: str
    row: Any


class FeaturesItem(TypedDict):
    dataset: str
    config: str
    features: Any


# Content of endpoint responses


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


class InfosContent(TypedDict):
    infos: List[InfoItem]


class ConfigsContent(TypedDict):
    configs: List[ConfigItem]


class SplitsContent(TypedDict):
    splits: List[SplitItem]


class RowsContent(TypedDict):
    features: List[FeaturesItem]
    rows: List[RowItem]


class StatusErrorContent(TypedDict):
    status_code: int
    exception: str
    message: str
    cause: str
    cause_message: str


Content = Union[
    ConfigsContent,
    DatasetsContent,
    InfosContent,
    RowsContent,
    SplitsContent,
    StatusErrorContent,
]
