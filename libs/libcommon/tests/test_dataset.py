# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.dataset import check_support


@pytest.mark.real_dataset
def test_check_support() -> None:
    dataset = "glue"
    hf_endpoint = "https://huggingface.co"
    hf_token = None
    check_support(dataset, hf_endpoint, hf_token)
