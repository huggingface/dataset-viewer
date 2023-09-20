# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import shutil
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.hf_api import DatasetInfo
from libcommon.simple_cache import has_some_cache, upsert_response

from cache_maintenance.cache_cleaner import clean_cache


@pytest.mark.parametrize(
    "dataset_infos,minimun_supported_datasets,should_keep",
    [
        ([DatasetInfo(id="dataset")], 1, True),  # do not delete, dataset is still supported
        ([], 1000, True),  # do not delete, number of supported datasets is less than threshold
        ([], 0, False),  # delete dataset
    ],
)
def test_clean_cache(dataset_infos: list[DatasetInfo], minimun_supported_datasets: int, should_keep: bool) -> None:
    assets_directory = "/tmp/assets"
    cached_assets_directory = "/tmp/cached-assets"
    dataset = "dataset"
    os.makedirs(f"{assets_directory}/{dataset}", exist_ok=True)
    os.makedirs(f"{cached_assets_directory}/{dataset}", exist_ok=True)

    asset_file = Path(f"{assets_directory}/{dataset}/image.jpg")
    asset_file.touch()
    assert asset_file.is_file()

    cached_asset_file = Path(f"{cached_assets_directory}/{dataset}/image.jpg")
    cached_asset_file.touch()
    assert cached_asset_file.is_file()

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        content={"config_names": [{"dataset": dataset, "config": "config"}]},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind="config-split-names-from-streaming",
        dataset=dataset,
        config="config",
        content={"splits": [{"dataset": dataset, "config": "config", "split": "split"}]},
        http_status=HTTPStatus.OK,
    )
    assert has_some_cache(dataset=dataset)

    with patch("cache_maintenance.cache_cleaner.get_supported_dataset_infos", return_value=dataset_infos):
        with patch("cache_maintenance.cache_cleaner.MINIMUM_SUPPORTED_DATASETS", minimun_supported_datasets):
            clean_cache(
                hf_endpoint="hf_endpoint",
                hf_token="hf_token",
                assets_directory=assets_directory,
                cached_assets_directory=cached_assets_directory,
            )

    assert asset_file.is_file() == should_keep
    assert cached_asset_file.is_file() == should_keep
    assert has_some_cache(dataset=dataset) == should_keep

    shutil.rmtree(assets_directory, ignore_errors=True)
    shutil.rmtree(cached_assets_directory, ignore_errors=True)
