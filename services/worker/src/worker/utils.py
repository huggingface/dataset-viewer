# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Any, List, Mapping, Optional, TypedDict


class DatasetItem(TypedDict):
    dataset: str


class ConfigItem(DatasetItem):
    config: Optional[str]


class SplitItem(ConfigItem):
    split: Optional[str]


class SplitsList(TypedDict):
    splits: List[SplitItem]


class FailedConfigItem(ConfigItem):
    error: Mapping[str, Any]


class DatasetSplitNamesResponse(TypedDict):
    splits: List[SplitItem]
    pending: List[ConfigItem]
    failed: List[FailedConfigItem]
