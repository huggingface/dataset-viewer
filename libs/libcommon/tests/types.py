# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from dataclasses import dataclass
from typing import Any

from datasets import Dataset

from libcommon.url_signer import AssetUrlPath


@dataclass
class DatasetFixture:
    dataset: Dataset
    expected_feature_type: Any
    expected_cell: Any
    expected_asset_url_paths: list[AssetUrlPath]
    expected_num_asset_urls: int
