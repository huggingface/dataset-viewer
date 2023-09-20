# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.hf_api import DatasetInfo
from libcommon.simple_cache import has_some_cache, upsert_response

from cache_maintenance.cache_cleaner import clean_cache


def test_clean_cache() -> None:
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

    dataset_info = DatasetInfo(id=dataset)
    with patch("cache_maintenance.cache_cleaner.get_supported_dataset_infos", return_value=[dataset_info]):
        clean_cache(
            hf_endpoint="hf_endpoint",
            hf_token="hf_token",
            assets_directory=assets_directory,
            cached_assets_directory=cached_assets_directory,
        )

    assert asset_file.is_file()
    assert cached_asset_file.is_file()
    assert has_some_cache(dataset=dataset)

    with patch("cache_maintenance.cache_cleaner.get_supported_dataset_infos", return_value=[]):
        clean_cache(
            hf_endpoint="hf_endpoint",
            hf_token="hf_token",
            assets_directory=assets_directory,
            cached_assets_directory=cached_assets_directory,
        )

    assert not asset_file.is_file()
    assert not cached_asset_file.is_file()
    assert not has_some_cache(dataset=dataset)
    os.rmdir(assets_directory)
    os.rmdir(cached_assets_directory)
