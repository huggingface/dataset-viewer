# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import base64
import enum
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

import orjson


class Status(str, enum.Enum):
    WAITING = "waiting"
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class Priority(str, enum.Enum):
    NORMAL = "normal"
    LOW = "low"


class JobParams(TypedDict):
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]
    partition_start: Optional[int]
    partition_end: Optional[int]


class JobInfo(TypedDict):
    job_id: str
    type: str
    params: JobParams
    priority: Priority


class FlatJobInfo(TypedDict):
    job_id: str
    type: str
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]
    priority: str


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        # see https://stackoverflow.com/a/40000564/7351594 for example
        # the bytes are encoded with base64, and then decoded as utf-8
        # (ascii only, by the way) to get a string
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


def orjson_dumps(content: Any) -> bytes:
    return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)


def get_datetime() -> datetime:
    return datetime.now(timezone.utc)


def inputs_to_string(
    dataset: str,
    revision: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    partition_start: Optional[int] = None,
    parition_end: Optional[int] = None,
    prefix: Optional[str] = None,
) -> str:
    result = dataset
    if revision is not None:
        result = f"{result},{revision}"
    if config is not None:
        result = f"{result},{config}"
        if split is not None:
            result = f"{result},{split}"
            if partition_start is not None:
                result = f"{result},{partition_start}"
                if parition_end is not None:
                    result = f"{result},{parition_end}"
    if prefix is not None:
        result = f"{prefix},{result}"
    return result
