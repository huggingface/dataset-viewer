import base64
import functools
import time
from distutils.util import strtobool
from logging import Logger
from os import _Environ
from typing import Any, Dict, List, Union

import orjson
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


def get_str_list_value(d: GenericDict, key: str, default: List[str]) -> List[str]:
    if key not in d:
        return default
    return [el.strip() for el in str(d.get(key)).split(",") if len(el.strip())]


def get_str_or_none_value(d: GenericDict, key: str, default: Union[str, None]) -> Union[str, None]:
    if key not in d:
        return default
    value = str(d.get(key)).strip()
    return default if value == "" else value


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


def orjson_dumps(content: Any) -> bytes:
    return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)


def retry(logger: Logger):
    def decorator_retry(func):
        """retries with an increasing sleep before every attempt"""
        SLEEPS = [1, 7, 70, 7 * 60, 70 * 60]
        MAX_ATTEMPTS = len(SLEEPS)

        @functools.wraps(func)
        def decorator(*args, **kwargs):
            attempt = 0
            while attempt < MAX_ATTEMPTS:
                try:
                    """always sleep before calling the function. It will prevent rate limiting in the first place"""
                    duration = SLEEPS[attempt]
                    logger.info(f"Sleep during {duration} seconds to preventively mitigate rate limiting.")
                    time.sleep(duration)
                    return func(*args, **kwargs)
                except ConnectionError:
                    logger.info("Got a ConnectionError, possibly due to rate limiting. Let's retry.")
                    attempt += 1
            raise Exception(f"Give up after {attempt} attempts with ConnectionError")

        return decorator

    return decorator_retry
