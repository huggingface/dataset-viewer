# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pytest
from libapi.exceptions import UnexpectedApiError
from libcommon.simple_cache import has_some_cache, upsert_response
from libcommon.storage_client import StorageClient
from pytest import raises

from admin.routes.obsolete_cache import (
    DatasetCacheReport,
    delete_obsolete_cache,
    get_obsolete_cache,
)

REVISION_NAME = "revision"


@pytest.mark.parametrize(
    "dataset_names,expected_report",
    [(["dataset"], []), ([], [DatasetCacheReport(dataset="dataset", cache_records=2)])],
)
def test_get_obsolete_cache(dataset_names: list[str], expected_report: list[DatasetCacheReport]) -> None:
    dataset = "dataset"

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": dataset, "config": "config"}]},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind="config-split-names-from-streaming",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
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
    dataset_names: list[str], minimun_supported_datasets: int, should_keep: bool, should_raise: bool, tmp_path: Path
) -> None:
    dataset = "dataset"

    assets_storage_client = StorageClient(
        protocol="file",
        root=str(tmp_path),
        folder="assets",
    )
    cached_assets_storage_client = StorageClient(
        protocol="file",
        root=str(tmp_path),
        folder="cached-assets",
    )

    cached_assets_storage_client._fs.mkdirs(dataset, exist_ok=True)
    assets_storage_client._fs.mkdirs(dataset, exist_ok=True)

    image_key = f"{dataset}/image.jpg"

    assets_storage_client._fs.touch(f"{assets_storage_client.get_base_directory()}/{image_key}")
    assert assets_storage_client.exists(image_key)

    cached_assets_storage_client._fs.touch(f"{cached_assets_storage_client.get_base_directory()}/{image_key}")
    assert cached_assets_storage_client.exists(image_key)

    upsert_response(
        kind="kind_1",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": dataset, "config": "config"}]},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind="kind_2",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config="config",
        content={"splits": [{"dataset": dataset, "config": "config", "split": "split"}]},
        http_status=HTTPStatus.OK,
    )
    assert has_some_cache(dataset=dataset)

    with patch("admin.routes.obsolete_cache.get_supported_dataset_names", return_value=dataset_names):
        with patch("admin.routes.obsolete_cache.MINIMUM_SUPPORTED_DATASETS", minimun_supported_datasets):
            if should_raise:
                with raises(UnexpectedApiError):
                    delete_obsolete_cache(
                        hf_endpoint="hf_endpoint",
                        hf_token="hf_token",
                        cached_assets_storage_client=cached_assets_storage_client,
                        assets_storage_client=assets_storage_client,
                    )
            else:
                deletion_report = delete_obsolete_cache(
                    hf_endpoint="hf_endpoint",
                    hf_token="hf_token",
                    cached_assets_storage_client=cached_assets_storage_client,
                    assets_storage_client=assets_storage_client,
                )
                assert len(deletion_report) == 0 if should_keep else 1
                if len(deletion_report) > 0:
                    assert deletion_report[0]["dataset"] == "dataset"
                    assert deletion_report[0]["cache_records"] == 2  # for kind_1 and kind_2
    assert assets_storage_client.exists(image_key) == should_keep
    assert cached_assets_storage_client.exists(image_key) == should_keep

    assert has_some_cache(dataset=dataset) == should_keep
