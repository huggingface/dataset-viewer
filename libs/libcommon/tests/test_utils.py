# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from libcommon.exceptions import DatasetInBlockListError
from libcommon.utils import (
    get_expires,
    inputs_to_string,
    is_image_url,
    orjson_dumps,
    raise_if_blocked,
)


@pytest.mark.parametrize(
    "dataset,revision,config,split,prefix,expected",
    [
        ("dataset", "revision", None, None, None, "dataset,revision"),
        ("dataset", "revision", "config", None, None, "dataset,revision,config"),
        ("dataset", "revision", None, "split", None, "dataset,revision"),
        ("dataset", "revision", "config", "split", None, "dataset,revision,config,split"),
        ("dataset", "revision", "config", "split", "prefix", "prefix,dataset,revision,config,split"),
    ],
)
def test_inputs_to_string(dataset: str, revision: str, config: str, split: str, prefix: str, expected: str) -> None:
    result = inputs_to_string(dataset=dataset, revision=revision, config=config, split=split, prefix=prefix)
    assert result == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Some text", False),
        ("http://test", False),
        ("http://test/file.png", True),
        ("https://test/file.jpg", True),
    ],
)
def test_is_image_url(text: str, expected: bool) -> None:
    assert is_image_url(text=text) == expected


@pytest.mark.parametrize(
    "dataset,blocked,expectation",
    [
        ("public", [""], pytest.raises(ValueError)),
        ("public", ["a/b/c"], pytest.raises(ValueError)),
        ("public", ["pub*"], pytest.raises(ValueError)),
        ("public/test", ["*/test"], pytest.raises(ValueError)),
        ("public", ["*"], pytest.raises(ValueError)),
        ("public", ["*/*"], pytest.raises(ValueError)),
        ("public", ["**/*"], pytest.raises(ValueError)),
        ("public", ["public"], pytest.raises(DatasetInBlockListError)),
        ("public", ["public", "audio"], pytest.raises(DatasetInBlockListError)),
        ("public/test", ["public/*"], pytest.raises(DatasetInBlockListError)),
        ("public/test", ["public/test"], pytest.raises(DatasetInBlockListError)),
        ("public", ["audio"], does_not_raise()),
        ("public", [], does_not_raise()),
    ],
)
def test_raise_if_blocked(dataset: str, blocked: list[str], expectation: Any) -> None:
    with expectation:
        raise_if_blocked(dataset=dataset, blocked_datasets=blocked)


def test_orjson_dumps() -> None:
    obj = {
        "int": 1,
        "bool": True,
        "str": "text",
        "np_array": np.array([1, 2, 3, 4]),
        "np_int64": np.int64(12),
        "pd_timedelta": pd.Timedelta(1, "d"),
        "object": {"a": 1, "b": 10.2},
        "non_string_key": {1: 20},
        "pd_timestmap": pd.Timestamp("2023-10-06"),
    }
    serialized_obj = orjson_dumps(obj)
    assert serialized_obj is not None


# see https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
EXAMPLE_DATETIME = datetime(2013, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
EXAMPLE_DATETIME_MINUS_ONE_HOUR = datetime(2013, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def test_get_expires() -> None:
    # see https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
    with patch("libcommon.utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = EXAMPLE_DATETIME_MINUS_ONE_HOUR
        # ^ 1 hour before the example date
        date = get_expires(seconds=3600)
        # check tests
        assert date > mock_datetime.now(timezone.utc)
        assert date < mock_datetime.now(timezone.utc) + timedelta(hours=2)
    assert date == EXAMPLE_DATETIME
