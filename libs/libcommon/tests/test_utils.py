# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.storage import StrPath
from libcommon.utils import (
    get_download_folder_for_split,
    inputs_to_string,
    is_image_url,
)


@pytest.mark.parametrize(
    "dataset,revision,config,split,prefix,expected",
    [
        ("dataset", None, None, None, None, "dataset"),
        ("dataset", "revision", None, None, None, "dataset,revision"),
        ("dataset", "revision", "config", None, None, "dataset,revision,config"),
        ("dataset", "revision", None, "split", None, "dataset,revision"),
        ("dataset", "revision", "config", "split", None, "dataset,revision,config,split"),
        ("dataset", None, "config", "split", None, "dataset,config,split"),
        ("dataset", None, None, None, "prefix", "prefix,dataset"),
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


def test_get_index_folder(cached_assets_directory: StrPath) -> None:
    dataset, config, split = "dataset", "config", "split"
    index_folder = get_download_folder_for_split(cached_assets_directory, dataset, config, split)
    assert index_folder is not None
    assert str(cached_assets_directory) in index_folder
