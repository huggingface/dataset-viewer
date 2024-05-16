import os.path
from collections.abc import Mapping
from pathlib import Path

import pytest
import validators  # type: ignore
from PIL import Image as PILImage

from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.asset import create_audio_file, create_image_file, generate_object_key

from ..constants import (
    ASSETS_BASE_URL,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_ROW_IDX,
    DEFAULT_SPLIT,
)
from ..types import DatasetFixture


def test_create_image_file(storage_client: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture]) -> None:
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
        storage_client=storage_client,
    )
    assert value == dataset_fixture.expected_cell
    image_key = value["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    assert storage_client.exists(image_key)

    image = PILImage.open(storage_client.get_full_path(image_key))
    assert image is not None


@pytest.mark.parametrize("audio_file_extension", ["WITH_EXTENSION", "WITHOUT_EXTENSION"])
@pytest.mark.parametrize("audio_file_name", ["test_audio_44100.wav", "test_audio_opus.opus"])
def test_create_audio_file(
    audio_file_name: str, audio_file_extension: str, shared_datadir: Path, storage_client: StorageClient
) -> None:
    audio_file_extension = os.path.splitext(audio_file_name)[1] if audio_file_extension == "WITH_EXTENSION" else ""
    audio_file_bytes = (shared_datadir / audio_file_name).read_bytes()
    value = create_audio_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        audio_file_extension=audio_file_extension,
        audio_file_bytes=audio_file_bytes,
        column="col",
        filename="audio.wav",
        storage_client=storage_client,
    )
    audio_key = "dataset/--/revision/--/config/split/7/col/audio.wav"
    assert value == [
        {
            "src": f"{ASSETS_BASE_URL}/{audio_key}",
            "type": "audio/wav",
        },
    ]
    assert storage_client.exists(audio_key)


@pytest.mark.parametrize(
    "dataset,config,split,column",
    [
        ("dataset", "config", "split", "column"),
        ("dataset", "config?<script>alert('XSS');</script>&", "split", "column?"),
    ],
)
def test_src_is_sanitized(dataset: str, config: str, split: str, column: str) -> None:
    base_url = "https://datasets-server.huggingface.co/assets"
    filename = "image.jpg"
    object_key = generate_object_key(
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
