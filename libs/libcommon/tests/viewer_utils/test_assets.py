# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.storage import StrPath
from libcommon.viewer_utils.asset import DATASET_SEPARATOR, create_csv_file


def test_create_csv_file(assets_directory: StrPath) -> None:
    dataset, config, split = "dataset", "config", "split"
    headers = ["col1", "col2"]
    data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]
    file_name = "dummy.csv"
    assets_base_url = "http://localhost"
    source = create_csv_file(
        assets_directory=assets_directory,
        assets_base_url=assets_base_url,
        dataset=dataset,
        config=config,
        split=split,
        headers=headers,
        data=data,
        file_name=file_name,
    )
    assert source
    assert isinstance(source, dict)
    assert source["src"] is not None
    assert source["src"] == f"{assets_base_url}/{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{file_name}"
    assert source["type"] is not None
    assert source["type"] == "text/csv"
    # TODO: Validate file content