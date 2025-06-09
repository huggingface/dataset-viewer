# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from pathlib import Path

import pytest
from datasets import Dataset, Image, Pdf
from datasets.table import embed_table_storage
from libcommon.constants import ROW_IDX_COLUMN
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import URLPreparator
from pdfplumber import open as open_pdf
from PIL import Image as PILImage

from libapi.response import create_response

pytestmark = pytest.mark.anyio

CACHED_ASSETS_FOLDER = "cached-assets"


@pytest.fixture
def storage_client(tmp_path: Path, hf_endpoint: str) -> StorageClient:
    return StorageClient(
        protocol="file",
        storage_root=str(tmp_path / CACHED_ASSETS_FOLDER),
        base_url="http://localhost/cached-assets",
        url_preparator=URLPreparator(
            url_signer=None, hf_endpoint=hf_endpoint, assets_base_url="http://localhost/cached-assets"
        ),
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


async def test_create_response_with_row_idx_column(storage_client: StorageClient) -> None:
    ds = Dataset.from_dict({"text": ["Hello there", "General Kenobi"], ROW_IDX_COLUMN: [3, 4]})
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
        use_row_idx_column=True,
    )
    assert response["features"] == [{"feature_idx": 0, "name": "text", "type": {"dtype": "string", "_type": "Value"}}]
    assert response["rows"] == [
        {"row_idx": 3, "row": {"text": "Hello there"}, "truncated_cells": []},
        {"row_idx": 4, "row": {"text": "General Kenobi"}, "truncated_cells": []},
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


async def test_create_response_with_document(document_path: str, storage_client: StorageClient) -> None:
    document_list = [document_path] * 5  # testing with multiple documents to ensure multithreading works
    ds = Dataset.from_dict({"document": document_list}).cast_column("document", Pdf())
    ds_document = Dataset(embed_table_storage(ds.data))
    dataset, config, split = "ds_document", "default", "train"
    document_key = "ds_document/--/revision/--/default/train/0/document/document.pdf"
    thumbnail_key = "ds_document/--/revision/--/default/train/0/document/document.pdf.png"

    response = await create_response(
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        storage_client=storage_client,
        pa_table=ds_document.data,
        offset=0,
        features=ds_document.features,
        unsupported_columns=[],
        num_rows_total=10,
        partial=False,
    )
    assert response["features"] == [{"feature_idx": 0, "name": "document", "type": {"_type": "Pdf"}}]
    assert len(response["rows"]) == len(document_list)
    assert response["partial"] is False
    assert storage_client.exists(document_key)
    pdf = open_pdf(f"{storage_client.storage_root}/{document_key}")
    assert pdf is not None
    thumbnail = PILImage.open(f"{storage_client.storage_root}/{thumbnail_key}")
    assert thumbnail is not None
