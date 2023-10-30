from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import pytest
import soundfile  # type: ignore
from datasets import Dataset
from PIL import Image as PILImage  # type: ignore

from libcommon.public_assets_storage import PublicAssetsStorage
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.asset import create_audio_file, create_image_file

ASSETS_FOLDER = "assets"
ASSETS_BASE_URL = f"http://localhost/{ASSETS_FOLDER}"


@pytest.fixture
def public_assets_storage(tmp_path: Path) -> PublicAssetsStorage:
    storage_client = StorageClient(
        protocol="file",
        root=str(tmp_path),
        folder=ASSETS_FOLDER,
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


def test_create_audio_file(datasets: Mapping[str, Dataset], public_assets_storage: PublicAssetsStorage) -> None:
    dataset = datasets["audio"]
    value = dataset[0]["col"]
    buffer = BytesIO()
    soundfile.write(buffer, value["array"], value["sampling_rate"], format="wav")
    audio_file_bytes = buffer.read()
    value = create_audio_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        audio_file_extension=".wav",
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
