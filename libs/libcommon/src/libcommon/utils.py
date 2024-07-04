# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import base64
import functools
import logging
import mimetypes
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, cast

import orjson
import pandas as pd
import pytz
from huggingface_hub import constants, hf_hub_download
from requests.exceptions import ReadTimeout

from libcommon.exceptions import DatasetInBlockListError


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        # see https://stackoverflow.com/a/40000564/7351594 for example
        # the bytes are encoded with base64, and then decoded as utf-8
        # (ascii only, by the way) to get a string
        return base64.b64encode(obj).decode("utf-8")
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    return str(obj)


def orjson_dumps(content: Any) -> bytes:
    return orjson.dumps(
        content, option=orjson.OPT_UTC_Z | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS, default=orjson_default
    )


def get_json_size(obj: Any) -> int:
    """Returns the size of an object in bytes once serialized as JSON

    Args:
        obj (`Any`): the Python object

    Returns:
        `int`: the size of the serialized object in bytes
    """
    return len(orjson_dumps(obj))


# from https://stackoverflow.com/a/43848928/7351594
def utf8_lead_byte(b: int) -> bool:
    """A UTF-8 intermediate byte starts with the bits 10xxxxxx."""
    return (b & 0xC0) != 0x80


class SmallerThanMaxBytesError(Exception):
    pass


def serialize_and_truncate(obj: Any, max_bytes: int) -> str:
    """
    Serialize an object as JSON and truncate it if it's bigger than max_bytes.

    Args:
        obj (`Any`): the Python object (can be a primitive type, a list, a dict, etc.)
        max_bytes (`int`): the maximum number of bytes.

    Raises:
        [`SmallerThanMaxBytesError`]: if the serialized object is smaller than max_bytes.

    Returns:
        `str`: the serialized object, truncated to max_bytes.
    """
    serialized_bytes = orjson_dumps(obj)
    if len(serialized_bytes) <= max_bytes:
        raise SmallerThanMaxBytesError()
    # If text[max_bytes] is not a lead byte, back up until a lead byte is
    # found and truncate before that character
    i = max_bytes
    while i > 0 and not utf8_lead_byte(serialized_bytes[i]):
        i -= 1
    return serialized_bytes[:i].decode("utf8", "ignore")


def get_datetime(days: Optional[float] = None) -> datetime:
    date = datetime.now(timezone.utc)
    if days is not None:
        date = date - timedelta(days=days)
    return date


def get_duration(started_at: datetime) -> int:
    """
    Get time in seconds that has passed from `started_at` until now. `started_at` must be in UTC time zone.
    `started_at` must be in UTC timezone.
    """
    return int((get_datetime() - pytz.UTC.localize(started_at)).total_seconds())


def get_expires(seconds: float) -> datetime:
    # see https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def inputs_to_string(
    dataset: str,
    revision: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    result = f"{dataset},{revision}"
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


def raise_if_blocked(
    dataset: str,
    blocked_datasets: list[str],
) -> None:
    """
    Raise an error if the dataset is in the list of blocked datasets

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        blocked_datasets (`list[str]`):
            The list of blocked datasets. If empty, no dataset is blocked.
            Unix shell-style wildcards are supported in the dataset name, e.g. "open-llm-leaderboard/*"
            to block all the datasets in the `open-llm-leaderboard` namespace. They are not allowed in
            the namespace name.

    Raises:
        [~`libcommon.exceptions.DatasetInBlockListError`]:
          If the dataset is in the list of blocked datasets.
    """
    for blocked_dataset in blocked_datasets:
        parts = blocked_dataset.split("/")
        if len(parts) > 2 or not blocked_dataset:
            raise ValueError(
                "The dataset name is not valid. It should be a namespace (user or an organization) and a repo name"
                " separated by a `/`, or a simple repo name for canonical datasets."
            )
        if "*" in parts[0]:
            raise ValueError("The namespace name, or the canonical dataset name, cannot contain a wildcard.")
        if fnmatch(dataset, blocked_dataset):
            raise DatasetInBlockListError(
                "This dataset has been disabled for now. Please open an issue in"
                " https://github.com/huggingface/dataset-viewer if you want this dataset to be supported."
            )


FuncT = TypeVar("FuncT", bound=Callable[..., Any])
RETRY_SLEEPS = (1, 1, 1, 10, 10, 10, 60, 60, 60, 10 * 60)
RETRY_ON: tuple[type[Exception]] = (Exception,)


class retry:
    """retries with an increasing sleep before every attempt"""

    def __init__(self, sleeps: Sequence[float] = RETRY_SLEEPS, on: Sequence[type[Exception]] = RETRY_ON) -> None:
        self.sleeps = sleeps
        self.on = on

    def __call__(self, func: FuncT) -> FuncT:
        @functools.wraps(func)
        def decorator(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            last_err = None
            while attempt < len(self.sleeps):
                try:
                    """always sleep before calling the function. It will prevent rate limiting in the first place"""
                    duration = self.sleeps[attempt]
                    logging.debug(f"Sleep during {duration} seconds to preventively mitigate rate limiting.")
                    time.sleep(duration)
                    return func(*args, **kwargs)
                except tuple(self.on) as err:
                    logging.info(f"Got a {type(err).__name__}. Let's retry.")
                    last_err = err
                    attempt += 1
            raise RuntimeError(f"Give up after {attempt} attempts. The last one raised {type(last_err)}") from last_err

        return cast(FuncT, decorator)


HF_HUB_HTTP_ERROR_RETRY_SLEEPS = [1, 1, 1, 10, 10, 10]


def download_file_from_hub(
    repo_type: str,
    revision: str,
    repo_id: str,
    filename: str,
    local_dir: Union[str, Path],
    hf_token: Optional[str],
    cache_dir: Union[str, Path, None] = None,
    force_download: bool = False,
    resume_download: bool = False,
) -> None:
    logging.debug(f"Using {constants.HF_HUB_ENABLE_HF_TRANSFER} for hf_transfer")
    retry_on = [RuntimeError] if constants.HF_HUB_ENABLE_HF_TRANSFER else [ReadTimeout]
    retry_download_hub_file = retry(on=retry_on, sleeps=HF_HUB_HTTP_ERROR_RETRY_SLEEPS)(hf_hub_download)
    retry_download_hub_file(
        repo_type=repo_type,
        revision=revision,
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
        force_download=force_download,
        cache_dir=cache_dir,
        resume_download=resume_download,
    )
