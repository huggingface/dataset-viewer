from typing import Any, Dict, List, TypedDict, Union


class ConfigsDict(TypedDict):
    dataset: str
    configs: List[str]


class InfoDict(TypedDict):
    dataset: str
    info: Dict


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


ResponseContent = Union[ConfigsDict, InfoDict, SplitsDict, RowsDict, StatusErrorDict]


class ResponseJSON(TypedDict):
    content: bytes
    status_code: int
