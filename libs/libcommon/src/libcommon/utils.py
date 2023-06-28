# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import base64
import enum
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Mapping, Optional, TypedDict

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
    status: str
    created_at: datetime


class JobOutput(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus
    error_code: Optional[str]
    details: Optional[Mapping[str, Any]]
    progress: Optional[float]


class JobResult(TypedDict):
    job_info: JobInfo
    job_runner_version: int
    is_success: bool
    output: Optional[JobOutput]


class SplitHubFile(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


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
    prefix: Optional[str] = None,
) -> str:
    result = dataset
    if revision is not None:
        result = f"{result},{revision}"
    if config is not None:
        result = f"{result},{config}"
        if split is not None:
            result = f"{result},{split}"
    if prefix is not None:
        result = f"{prefix},{result}"
    return result


def is_image_url(text: str) -> bool:
    is_url = text.startswith("https://") or text.startswith("http://")
    (mime_type, _) = mimetypes.guess_type(text.split("/")[-1].split("?")[0])
    return is_url and mime_type is not None and mime_type.startswith("image/")
