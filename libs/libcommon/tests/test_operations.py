# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.operations import get_dataset_status


@pytest.mark.real_dataset
def test_get_dataset_status() -> None:
    dataset = "glue"
    hf_endpoint = "https://huggingface.co"
    hf_token = None
    get_dataset_status(dataset, hf_endpoint, hf_token)


@pytest.mark.real_dataset
def test_get_dataset_status_timeout() -> None:
    dataset = "glue"
    hf_endpoint = "https://huggingface.co"
    hf_token = None
    with pytest.raises(Exception):
        get_dataset_status(dataset, hf_endpoint, hf_token, hf_timeout_seconds=0.01)
