from typing import Any, Dict, List, TypedDict, Union


class DatasetsDict(TypedDict):
    datasets: List[str]


class ConfigsDict(TypedDict):
    dataset: str
    configs: List[str]


class InfoDict(TypedDict):
    dataset: str
    info: Dict[str, Any]


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


ResponseContent = Union[DatasetsDict, ConfigsDict, InfoDict, SplitsDict, RowsDict, StatusErrorDict]


class Report(TypedDict):
    success: bool
    exception: Union[str, None]
    message: Union[str, None]
    cause: Union[str, None]
    cause_message: Union[str, None]
    elapsed_seconds: float


class InfoReport(Report):
    dataset: str
    info_num_keys: Union[int, None]


class ConfigsReport(Report):
    dataset: str
    configs: List[str]


class SplitsReport(Report):
    dataset: str
    config: str
    splits: List[str]


class RowsReport(Report):
    dataset: str
    config: str
    split: str
    rows_length: Union[int, None]
