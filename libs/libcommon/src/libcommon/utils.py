# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import base64
import mimetypes
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from typing import Any, Optional

import orjson
import pandas as pd

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


def get_datetime(days: Optional[float] = None) -> datetime:
    date = datetime.now(timezone.utc)
    if days is not None:
        date = date - timedelta(days=days)
    return date


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

    Returns:
        `None`
    Raises the following errors:
        - [`libcommon.exceptions.DatasetInBlockListError`]
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
                " https://github.com/huggingface/datasets-server if you want this dataset to be supported."
            )
