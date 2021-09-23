from typing import Any, Dict, List, TypedDict, Union


class DatasetItem(TypedDict):
    dataset: str


class DatasetsDict(TypedDict):
    datasets: List[DatasetItem]


class InfoItem(TypedDict):
    dataset: str
    info: Dict[str, Any]


InfoDict = InfoItem


class ConfigsDict(TypedDict):
    dataset: str
    configs: List[str]


class SplitsDict(TypedDict):
    dataset: str
    config: str
    splits: List[str]


class RowsDict(TypedDict):
    dataset: str
    config: str
    split: str
    rows: List[Any]


class StatusErrorDict(TypedDict):
    status_code: int
    exception: str
    message: str
    cause: str
    cause_message: str


ResponseContent = Union[DatasetsDict, InfoDict, ConfigsDict, SplitsDict, RowsDict, StatusErrorDict]
