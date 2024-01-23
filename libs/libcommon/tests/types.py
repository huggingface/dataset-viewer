# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from dataclasses import dataclass
from typing import Any

from datasets import Dataset


@dataclass
class DatasetFixture:
    dataset: Dataset
    expected_feature_type: Any
    expected_row: Any
