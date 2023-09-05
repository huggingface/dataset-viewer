# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from .fixtures.hub import DatasetRepos
from .utils import get, get_default_config_split


def test_is_valid_after_datasets_processed(hf_dataset_repos_csv_data: DatasetRepos) -> None:
    # this test ensures that a dataset processed successfully returns true in /is-valid
    response = get("/is-valid")
    assert response.status_code == 422, f"{response.status_code} - {response.text}"
    # at this moment various datasets have been processed (due to the alphabetic order of the test files)
    public = hf_dataset_repos_csv_data["public"]
    config, split = get_default_config_split()

    split_response = get(f"/is-valid?dataset={public}&config={config}&split={split}")
    assert split_response.status_code == 200, f"{split_response.status_code} - {split_response.text}"

    config_response = get(f"/is-valid?dataset={public}&config={config}")
    assert config_response.status_code == 200, f"{config_response.status_code} - {config_response.text}"

    response = get(f"/is-valid?dataset={public}")
    assert response.status_code == 200, f"{response.status_code} - {response.text}"

    # TODO: re-enable once https://github.com/huggingface/datasets-server/issues/891 is done
    # For now: /valid is not coherent, it only relies on having a success response for split-first-rows-from-streaming
    # which is not the case when split-first-rows-from-dataset-info is a success (parallel steps)

    # assert response.json()["valid"], response.text
    # # without authentication, we get a 401 error when requesting a non-existing dataset
    # response = get("/is-valid?dataset=non-existing-dataset")
    # assert response.status_code == 401, f"{response.status_code} - {response.text}"
