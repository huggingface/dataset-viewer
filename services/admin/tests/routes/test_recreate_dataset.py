# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import shutil
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from libcommon.dtos import Priority, Status
from libcommon.queue.jobs import Queue
from libcommon.simple_cache import has_some_cache, upsert_response
from libcommon.storage_client import StorageClient

from admin.routes.recreate_dataset import recreate_dataset

REVISION_NAME = "revision"


def test_recreate_dataset(tmp_path: Path) -> None:
    assets_directory = tmp_path / "assets"
    cached_assets_directory = tmp_path / "cached-assets"

    assets_storage_client = StorageClient(
        protocol="file",
        storage_root=str(assets_directory),
        base_url="http://notimportant/assets",
    )
    cached_assets_storage_client = StorageClient(
        protocol="file",
        storage_root=str(cached_assets_directory),
        base_url="http://notimportant/cached-assets",
    )

    dataset = "dataset"
    os.makedirs(f"{assets_directory}/{dataset}", exist_ok=True)
    os.makedirs(f"{cached_assets_directory}/{dataset}", exist_ok=True)

    asset_file = assets_directory / dataset / "image.jpg"
    asset_file.touch()
    assert asset_file.is_file()

    cached_asset_file = cached_assets_directory / dataset / "image.jpg"
    cached_asset_file.touch()
    assert cached_asset_file.is_file()

    queue = Queue()
    queue.add_job(
        job_type="job_type",
        dataset=dataset,
        revision=REVISION_NAME,
        config=None,
        split=None,
        difficulty=100,
    ).update(status=Status.STARTED)

    upsert_response(
        kind="kind",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": dataset, "config": "config"}]},
        http_status=HTTPStatus.OK,
    )

    assert has_some_cache(dataset=dataset)

    with patch("admin.routes.recreate_dataset.update_dataset", return_value=None):
        recreate_dataset_report = recreate_dataset(
            hf_endpoint="hf_endpoint",
            hf_token="hf_token",
            dataset=dataset,
            priority=Priority.HIGH,
            blocked_datasets=[],
            storage_clients=[assets_storage_client, cached_assets_storage_client],
        )
        assert recreate_dataset_report["status"] == "ok"
        assert recreate_dataset_report["dataset"] == dataset
    assert not asset_file.is_file()
    assert not cached_asset_file.is_file()
    assert not has_some_cache(dataset=dataset)

    shutil.rmtree(assets_directory, ignore_errors=True)
    shutil.rmtree(cached_assets_directory, ignore_errors=True)
