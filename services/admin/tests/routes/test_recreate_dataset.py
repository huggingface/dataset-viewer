# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import shutil
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import has_some_cache, upsert_response
from libcommon.utils import Priority, Status

from admin.routes.recreate_dataset import recreate_dataset

REVISION_NAME = "revision"


def test_recreate_dataset(tmp_path: Path, processing_graph: ProcessingGraph) -> None:
    assets_directory = tmp_path / "assets"
    cached_assets_directory = tmp_path / "cached-assets"
    dataset = "dataset"
    os.makedirs(f"{assets_directory}/{dataset}", exist_ok=True)
    os.makedirs(f"{cached_assets_directory}/{dataset}", exist_ok=True)

    asset_file = Path(f"{assets_directory}/{dataset}/image.jpg")
    asset_file.touch()
    assert asset_file.is_file()

    cached_asset_file = Path(f"{cached_assets_directory}/{dataset}/image.jpg")
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

    with patch("admin.routes.recreate_dataset.get_dataset_git_revision", return_value=REVISION_NAME):
        recreate_dataset_report = recreate_dataset(
            hf_endpoint="hf_endpoint",
            hf_token="hf_token",
            assets_directory=assets_directory,
            cached_assets_directory=cached_assets_directory,
            dataset=dataset,
            priority=Priority.HIGH,
            processing_graph=processing_graph,
            blocked_datasets=[],
        )
        assert recreate_dataset_report["status"] == "ok"
        assert recreate_dataset_report["dataset"] == dataset
        assert recreate_dataset_report["cancelled_jobs"] == 1
        assert recreate_dataset_report["deleted_cached_responses"] == 1
    assert not asset_file.is_file()
    assert not cached_asset_file.is_file()
    assert not has_some_cache(dataset=dataset)

    shutil.rmtree(assets_directory, ignore_errors=True)
    shutil.rmtree(cached_assets_directory, ignore_errors=True)
