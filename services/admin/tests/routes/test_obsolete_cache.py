# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import shutil
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pytest
from libcommon.simple_cache import has_some_cache, upsert_response
from pytest import raises

from admin.routes.obsolete_cache import (
    DatasetCacheReport,
    delete_obsolete_cache,
    get_obsolete_cache,
)
from admin.utils import UnexpectedError


@pytest.mark.parametrize(
    "dataset_names,expected_report",
    [(["dataset"], []), ([], [DatasetCacheReport(dataset="dataset", cache_records=2)])],
)
def test_get_obsolete_cache(dataset_names: list[str], expected_report: list[DatasetCacheReport]) -> None:
    dataset = "dataset"

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

    with patch("admin.routes.obsolete_cache.get_supported_dataset_names", return_value=dataset_names):
        assert get_obsolete_cache(hf_endpoint="hf_endpoint", hf_token="hf_token") == expected_report


@pytest.mark.parametrize(
    "dataset_names,minimun_supported_datasets,should_keep,should_raise",
    [
        (["dataset"], 1, True, False),  # do not delete, dataset is still supported
        ([], 1000, True, True),  # do not delete, number of supported datasets is less than threshold
        ([], 0, False, False),  # delete dataset
    ],
)
def test_delete_obsolete_cache(
    dataset_names: list[str], minimun_supported_datasets: int, should_keep: bool, should_raise: bool
) -> None:
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
        kind="kind_1",
        dataset=dataset,
        content={"config_names": [{"dataset": dataset, "config": "config"}]},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind="kind_2",
        dataset=dataset,
        config="config",
        content={"splits": [{"dataset": dataset, "config": "config", "split": "split"}]},
        http_status=HTTPStatus.OK,
    )
    assert has_some_cache(dataset=dataset)

    with patch("admin.routes.obsolete_cache.get_supported_dataset_names", return_value=dataset_names):
        with patch("admin.routes.obsolete_cache.MINIMUM_SUPPORTED_DATASETS", minimun_supported_datasets):
            if should_raise:
                with raises(UnexpectedError):
                    delete_obsolete_cache(
                        hf_endpoint="hf_endpoint",
                        hf_token="hf_token",
                        assets_directory=assets_directory,
                        cached_assets_directory=cached_assets_directory,
                    )
            else:
                deletion_report = delete_obsolete_cache(
                    hf_endpoint="hf_endpoint",
                    hf_token="hf_token",
                    assets_directory=assets_directory,
                    cached_assets_directory=cached_assets_directory,
                )
                assert len(deletion_report) == 0 if should_keep else 1
                if len(deletion_report) > 0:
                    assert deletion_report[0]["dataset"] == "dataset"
                    assert deletion_report[0]["cache_records"] == 2  # for kind_1 and kind_2
    assert asset_file.is_file() == should_keep
    assert cached_asset_file.is_file() == should_keep
    assert has_some_cache(dataset=dataset) == should_keep

    shutil.rmtree(assets_directory, ignore_errors=True)
    shutil.rmtree(cached_assets_directory, ignore_errors=True)
