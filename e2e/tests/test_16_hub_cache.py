# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from .fixtures.hub import DatasetRepos
from .utils import get


def test_hub_cache_after_datasets_processed(hf_dataset_repos_csv_data: DatasetRepos) -> None:
    # this test ensures that the datasets processed successfully are present in /valid
    link = None
    for _ in range(10):
        if link is None:
            response = get(relative_url="/hub-cache")
        else:
            response = get(relative_url="", url=link)
        assert response.status_code == 200
        body = response.json()
        print(body)
        assert len(body) > 0
        assert all(item["viewer"] and item["num_rows"] > 0 for item in body)
        if "Link" not in response.headers:
            return
        link = response.headers["Link"].split(";")[0][1:-1]
        print(link)
