import os.path
from collections.abc import Mapping
from pathlib import Path

import pytest
import validators  # type: ignore
from pdfplumber import open
from PIL import Image as PILImage

from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.asset import (
    DATASET_GIT_REVISION_PLACEHOLDER,
    SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE,
    create_audio_file,
    create_image_file,
    create_pdf_file,
)

from ..constants import (
    ASSETS_BASE_URL,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_ROW_IDX,
    DEFAULT_SPLIT,
)
from ..types import DatasetFixture


def test_create_image_file(
    storage_client_with_url_preparator: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture]
) -> None:
    dataset_name = "image"
    dataset_fixture = datasets_fixtures[dataset_name]
    value = create_image_file(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        image=dataset_fixture.dataset[DEFAULT_ROW_IDX][DEFAULT_COLUMN_NAME],
        column=DEFAULT_COLUMN_NAME,
        filename="image.jpg",
        row_idx=DEFAULT_ROW_IDX,
        format="JPEG",
        storage_client=storage_client_with_url_preparator,
    )
    assert value == dataset_fixture.expected_cell
    image_key = value["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    image_path = image_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client_with_url_preparator.exists(image_path)

    image = PILImage.open(storage_client_with_url_preparator.get_full_path(image_path))
    assert image is not None


@pytest.mark.parametrize("audio_file_extension", ["WITH_EXTENSION", "WITHOUT_EXTENSION"])
@pytest.mark.parametrize("audio_file_name", ["test_audio_44100.wav", "test_audio_opus.opus"])
@pytest.mark.parametrize("filename_extension", [".wav", ".opus"])
def test_create_audio_file(
    audio_file_name: str,
    audio_file_extension: str,
    filename_extension: str,
    shared_datadir: Path,
    storage_client_with_url_preparator: StorageClient,
) -> None:
    audio_file_extension = os.path.splitext(audio_file_name)[1] if audio_file_extension == "WITH_EXTENSION" else ""
    audio_file_bytes = (shared_datadir / audio_file_name).read_bytes()
    filename = "audio" + filename_extension
    mime_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[filename_extension]
    value = create_audio_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        audio_file_extension=audio_file_extension,
        audio_file_bytes=audio_file_bytes,
        column="col",
        filename=filename,
        storage_client=storage_client_with_url_preparator,
    )
    audio_key = "dataset/--/revision/--/config/split/7/col/" + filename
    assert value == [
        {
            "src": f"{ASSETS_BASE_URL}/{audio_key}",
            "type": mime_type,
        },
    ]
    assert storage_client_with_url_preparator.exists(audio_key)


@pytest.mark.parametrize(
    "pdf_file,expected_size",
    [("test_A4.pdf", 8810), ("test_us_letter.pdf", 1319)],
)
def test_create_pdf_file(
    pdf_file: str,
    expected_size: int,
    shared_datadir: Path,
    storage_client_with_url_preparator: StorageClient,
) -> None:
    pdf = open(shared_datadir / pdf_file)
    value = create_pdf_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        column="col",
        filename=pdf_file,
        pdf=pdf,
        storage_client=storage_client_with_url_preparator,
    )
    pdf_key = value["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    pdf_path = pdf_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client_with_url_preparator.exists(pdf_path)
    new_pdf = open(storage_client_with_url_preparator.get_full_path(pdf_path))
    assert new_pdf is not None

    thumbnail_key = value["thumbnail"]["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    thumbnail_path = thumbnail_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client_with_url_preparator.exists(thumbnail_path)
    image = PILImage.open(storage_client_with_url_preparator.get_full_path(thumbnail_path))
    assert image is not None
    assert image.size == (value["thumbnail"]["width"], value["thumbnail"]["height"])
    assert value["size_bytes"] == expected_size


@pytest.mark.parametrize(
    "dataset,config,split,column",
    [
        ("dataset", "config", "split", "column"),
        ("dataset", "config?<script>alert('XSS');</script>&", "split", "column?"),
    ],
)
def test_src_is_sanitized(storage_client: StorageClient, dataset: str, config: str, split: str, column: str) -> None:
    base_url = "https://datasets-server.huggingface.co/assets"
    filename = "image.jpg"
    object_key = storage_client.generate_object_path(
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        row_idx=0,
        column=column,
        filename=filename,
    )
    src = f"{base_url}/{object_key}"
    assert validators.url(src)
