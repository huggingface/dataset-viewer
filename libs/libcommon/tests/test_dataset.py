# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import DatasetInfoHubRequestError


@pytest.mark.real_dataset
def test_get_dataset_git_revision() -> None:
    dataset = "glue"
    hf_endpoint = "https://huggingface.co"
    hf_token = None
    get_dataset_git_revision(dataset, hf_endpoint, hf_token)


@pytest.mark.real_dataset
def test_get_dataset_git_revision_timeout() -> None:
    dataset = "glue"
    hf_endpoint = "https://huggingface.co"
    hf_token = None
    with pytest.raises(DatasetInfoHubRequestError):
        get_dataset_git_revision(dataset, hf_endpoint, hf_token, hf_timeout_seconds=0.01)
