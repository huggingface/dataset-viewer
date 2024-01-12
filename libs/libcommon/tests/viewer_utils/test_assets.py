from collections.abc import Mapping
from pathlib import Path

import pytest
import validators  # type: ignore
from datasets import Dataset
from PIL import Image as PILImage  # type: ignore

from libcommon.public_assets_storage import PublicAssetsStorage
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.asset import create_audio_file, create_image_file, generate_asset_src

ASSETS_FOLDER = "assets"
ASSETS_BASE_URL = f"http://localhost/{ASSETS_FOLDER}"


@pytest.fixture
def public_assets_storage(tmp_path: Path) -> PublicAssetsStorage:
    storage_client = StorageClient(
        protocol="file",
        storage_root=str(tmp_path / ASSETS_FOLDER),
    )
    return PublicAssetsStorage(
        assets_base_url=ASSETS_BASE_URL,
        overwrite=False,
        storage_client=storage_client,
    )


def test_create_image_file(datasets: Mapping[str, Dataset], public_assets_storage: PublicAssetsStorage) -> None:
    dataset = datasets["image"]
    value = create_image_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        image=dataset[0]["col"],
        column="col",
        filename="image.jpg",
        row_idx=7,
        format="JPEG",
        public_assets_storage=public_assets_storage,
    )
    image_key = "dataset/--/revision/--/config/split/7/col/image.jpg"
    assert value == {
        "src": f"{ASSETS_BASE_URL}/{image_key}",
        "height": 480,
        "width": 640,
    }
    assert public_assets_storage.storage_client.exists(image_key)

    image = PILImage.open(f"{public_assets_storage.storage_client.get_base_directory()}/{image_key}")
    assert image is not None


@pytest.mark.parametrize("audio_file_extension", [".wav", ""])  # also test audio files without extension
def test_create_audio_file(
    audio_file_extension: str, shared_datadir: Path, public_assets_storage: PublicAssetsStorage
) -> None:
    audio_file_bytes = (shared_datadir / "test_audio_44100.wav").read_bytes()
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
        public_assets_storage=public_assets_storage,
    )

    audio_key = "dataset/--/revision/--/config/split/7/col/audio.wav"
    assert value == [
        {
            "src": f"{ASSETS_BASE_URL}/{audio_key}",
            "type": "audio/wav",
        },
    ]

    assert public_assets_storage.storage_client.exists(audio_key)


@pytest.mark.parametrize(
    "dataset,config,split,column",
    [
        ("dataset", "config", "split", "column"),
        ("dataset", "config?<script>alert('XSS');</script>&", "split", "column?"),
    ],
)
def test_generate_asset_src(dataset: str, config: str, split: str, column: str) -> None:
    base_url = "https://datasets-server.huggingface.co/assets"
    filename = "image.jpg"
    _, src = generate_asset_src(
        base_url=base_url,
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        row_idx=0,
        column=column,
        filename=filename,
    )
    assert validators.url(src)
