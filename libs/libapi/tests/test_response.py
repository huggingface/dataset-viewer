# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from pathlib import Path

import pytest
from datasets import Dataset, Image
from datasets.table import embed_table_storage
from libcommon.storage_client import StorageClient
from PIL import Image as PILImage  # type: ignore

from libapi.response import create_response

pytestmark = pytest.mark.anyio

CACHED_ASSETS_FOLDER = "cached-assets"


@pytest.fixture
def storage_client(tmp_path: Path) -> StorageClient:
    return StorageClient(
        protocol="file",
        storage_root=str(tmp_path / CACHED_ASSETS_FOLDER),
        base_url="http://localhost/cached-assets",
    )


async def test_create_response(storage_client: StorageClient) -> None:
    ds = Dataset.from_dict({"text": ["Hello there", "General Kenobi"]})
    response = await create_response(
        dataset="ds",
        revision="revision",
        config="default",
        split="train",
        storage_client=storage_client,
        pa_table=ds.data,
        offset=0,
        features=ds.features,
        unsupported_columns=[],
        num_rows_total=10,
        partial=False,
    )
    assert response["features"] == [{"feature_idx": 0, "name": "text", "type": {"dtype": "string", "_type": "Value"}}]
    assert response["rows"] == [
        {"row_idx": 0, "row": {"text": "Hello there"}, "truncated_cells": []},
        {"row_idx": 1, "row": {"text": "General Kenobi"}, "truncated_cells": []},
    ]
    assert response["num_rows_total"] == 10
    assert response["num_rows_per_page"] == 100
    assert response["partial"] is False


async def test_create_response_with_image(image_path: str, storage_client: StorageClient) -> None:
    ds = Dataset.from_dict({"image": [image_path]}).cast_column("image", Image())
    ds_image = Dataset(embed_table_storage(ds.data))
    dataset, config, split = "ds_image", "default", "train"
    image_key = "ds_image/--/revision/--/default/train/0/image/image.jpg"
    response = await create_response(
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        storage_client=storage_client,
        pa_table=ds_image.data,
        offset=0,
        features=ds_image.features,
        unsupported_columns=[],
        num_rows_total=10,
        partial=False,
    )
    assert response["features"] == [{"feature_idx": 0, "name": "image", "type": {"_type": "Image"}}]
    assert response["rows"] == [
        {
            "row_idx": 0,
            "row": {
                "image": {
                    "src": f"http://localhost/cached-assets/{image_key}",
                    "height": 480,
                    "width": 640,
                }
            },
            "truncated_cells": [],
        }
    ]
    assert response["partial"] is False
    assert storage_client.exists(image_key)
    image = PILImage.open(f"{storage_client.storage_root}/{image_key}")
    assert image is not None
