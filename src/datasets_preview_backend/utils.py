from distutils.util import strtobool
from os import _Environ
from typing import Dict, Union

from starlette.datastructures import QueryParams

GenericDict = Union[_Environ[str], QueryParams, Dict[str, Union[str, int, bool]]]


def get_bool_value(d: GenericDict, key: str, default: bool) -> bool:
    if key not in d:
        return default
    try:
        value = bool(strtobool(str(d.get(key))))
    except (TypeError, ValueError):
        value = default
    return value


def get_int_value(d: GenericDict, key: str, default: int) -> int:
    v = d.get(key)
    if v is None:
        return default
    try:
        value = int(v)
    except (TypeError, ValueError):
        value = default
    return value


def get_str_value(d: GenericDict, key: str, default: str) -> str:
    if key not in d:
        return default
    value = str(d.get(key)).strip()
    return default if value == "" else value


def get_str_or_none_value(d: GenericDict, key: str, default: Union[str, None]) -> Union[str, None]:
    if key not in d:
        return default
    value = str(d.get(key)).strip()
    return default if value == "" else value
