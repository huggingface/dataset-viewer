# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from .fixtures.hub import DatasetRepos
from .utils import get


def test_valid_after_datasets_processed(hf_dataset_repos_csv_data: DatasetRepos) -> None:
    # this test ensures that the datasets processed successfully are present in /valid
    response = get("/valid")
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    # at this moment various datasets have been processed (due to the alphabetic order of the test files)
    valid = response.json()["valid"]
    assert hf_dataset_repos_csv_data["public"] in valid, response.text
    assert hf_dataset_repos_csv_data["gated"] in valid, response.text
    assert hf_dataset_repos_csv_data["private"] not in valid, response.text
