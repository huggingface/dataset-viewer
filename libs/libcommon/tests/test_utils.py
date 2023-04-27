# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.utils import inputs_to_string


@pytest.mark.parametrize(
    "dataset,config,split,prefix,expected",
    [
        ("dataset", None, None, None, "dataset"),
        ("dataset", "config", None, None, "dataset,config"),
        ("dataset", None, "split", None, "dataset"),
        ("dataset", "config", "split", None, "dataset,config,split"),
        ("dataset", None, None, "prefix", "prefix,dataset"),
        ("dataset", "config", "split", "prefix", "prefix,dataset,config,split"),
    ],
)
def test_inputs_to_string(dataset: str, config: str, split: str, prefix: str, expected: str) -> None:
    result = inputs_to_string(dataset=dataset, config=config, split=split, prefix=prefix)
    assert result == expected
