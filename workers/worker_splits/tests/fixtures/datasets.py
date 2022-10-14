# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Any, Dict

import pytest
from datasets import Audio, Dataset, Features
from datasets.features.features import FeatureType


def other(content: Any, feature_type: FeatureType = None) -> Dataset:
    return (
        Dataset.from_dict({"col": [content]})
        if feature_type is None
        else Dataset.from_dict({"col": [content]}, features=Features({"col": feature_type}))
    )


@pytest.fixture(scope="session")
def datasets() -> Dict[str, Dataset]:
    sampling_rate = 16_000
    return {
        "audio": other({"array": [0.1, 0.2, 0.3], "sampling_rate": sampling_rate}, Audio(sampling_rate=sampling_rate)),
    }
