import os
import shutil
from collections.abc import Mapping
from io import BytesIO
from uuid import uuid4

import boto3
import soundfile  # type: ignore
from datasets import Dataset
from moto import mock_s3
from PIL import Image as PILImage  # type: ignore

from libcommon.storage_client import StorageClient
from libcommon.storage import StrPath
from libcommon.storage_options import S3StorageOptions
from libcommon.viewer_utils.asset import create_audio_file, create_image_file


def test_create_image_file_with_s3_storage(datasets: Mapping[str, Dataset], cached_assets_directory: StrPath) -> None:
    # ensure directory is emtpy
    assert len(os.listdir(cached_assets_directory)) == 0
    dataset = datasets["image"]

    bucket = str(uuid4())
    assets_folder = "assets"
    folder_name = f"/tmp/{bucket}/{assets_folder}"
    os.makedirs(folder_name)

    storage_client = StorageClient(implementation="file", root=f"/tmp/{bucket}")
    storage_options = S3StorageOptions(
        assets_base_url="http://localhost/assets",
        assets_directory=cached_assets_directory,
        overwrite=True,
        storage_client=storage_client,
        s3_folder_name=assets_folder,
    )

    value = create_image_file(
        dataset="dataset",
        config="config",
        split="split",
        image=dataset[0]["col"],
        column="col",
        filename="image.jpg",
        row_idx=7,
        storage_options=storage_options,
    )
    assert value == {
        "src": "http://localhost/assets/dataset/--/config/split/7/col/image.jpg",
        "height": 480,
        "width": 640,
    }
    assert storage_client.exists("assets/dataset/--/config/split/7/col/image.jpg")

    image = PILImage.open(f"{folder_name}/dataset/--/config/split/7/col/image.jpg")
    assert image is not None
    # ensure directory remains emtpy after file uploading
    assert len(os.listdir(cached_assets_directory)) == 0
    shutil.rmtree(f"/tmp/{bucket}", ignore_errors=True)


def test_create_audio_file_with_s3_storage(datasets: Mapping[str, Dataset], cached_assets_directory: StrPath) -> None:
    dataset = datasets["audio"]
    value = dataset[0]["col"]
    buffer = BytesIO()
    soundfile.write(buffer, value["array"], value["sampling_rate"], format="wav")
    audio_file_bytes = buffer.read()
    with mock_s3():
        bucket_name = "bucket"
        region = "us-east-1"
        access_key_id = "access_key_id"
        secret_access_key = "secret_access_key"
        folder_name = "assets"
        conn = boto3.resource("s3", region_name=region)
        conn.create_bucket(Bucket=bucket_name)

        storage_client = StorageClient(
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            bucket_name=bucket_name,
        )
        storage_options = S3StorageOptions(
            assets_base_url="http://localhost/assets",
            assets_directory=cached_assets_directory,
            overwrite=True,
            storage_client=storage_client,
            s3_folder_name=folder_name,
        )

        value = create_audio_file(
            dataset="dataset",
            config="config",
            split="split",
            row_idx=7,
            audio_file_extension=".wav",
            audio_file_bytes=audio_file_bytes,
            column="col",
            filename="audio.wav",
            storage_options=storage_options,
        )

        assert value == [
            {
                "src": "http://localhost/assets/dataset/--/config/split/7/col/audio.wav",
                "type": "audio/wav",
            },
        ]
        audio_object = conn.Object(bucket_name, "assets/dataset/--/config/split/7/col/audio.wav").get()["Body"].read()
        assert audio_object is not None
