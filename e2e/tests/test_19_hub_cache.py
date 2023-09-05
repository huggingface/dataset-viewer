# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os

from .fixtures.hub import DatasetRepos
from .utils import get


def test_hub_cache_after_datasets_processed(hf_dataset_repos_csv_data: DatasetRepos) -> None:
    # this test ensures that the datasets processed successfully are present in /hub-cache
    link = None
    iteration = 0
    keep_iterating = True
    while keep_iterating and iteration < 10:
        if link is None:
            response = get(relative_url="/hub-cache")
        else:
            response = get(url=link, relative_url="")
        assert response.status_code == 200
        body = response.json()

        keep_iterating = "Link" in response.headers
        link = response.headers["Link"].split(";")[0][1:-1] if keep_iterating else None
        iteration += 1

        print(f"{body=}")
        print(f"{response.headers=}")
        print(f"{link=}")

        assert all(item["viewer"] and item["num_rows"] > 0 for item in body)

    NUM_DATASETS = 4
    NUM_RESULTS_PER_PAGE = int(os.environ.get("HUB_CACHE_NUM_RESULTS_PER_PAGE", 1_000))
    assert iteration >= NUM_DATASETS // NUM_RESULTS_PER_PAGE
