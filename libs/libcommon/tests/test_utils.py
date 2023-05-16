# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.utils import inputs_to_string


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
