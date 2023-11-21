# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict
from urllib import parse

from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.constants import DATASET_SEPARATOR
from libcommon.public_assets_storage import PublicAssetsStorage
from libcommon.storage import StrPath, remove_dir

ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"
SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


def delete_asset_dir(dataset: str, directory: StrPath) -> None:
    dir_path = Path(directory).resolve() / dataset
    remove_dir(dir_path)


class ImageSource(TypedDict):
    src: str
    height: int
    width: int


class AudioSource(TypedDict):
    src: str
    type: str


def generate_asset_src(
    base_url: str, dataset: str, revision: str, config: str, split: str, row_idx: int, column: str, filename: str
) -> tuple[str, str]:
    dir_path = f"{parse.quote(dataset)}/{DATASET_SEPARATOR}/{revision}/{DATASET_SEPARATOR}/{parse.quote(config)}/{parse.quote(split)}/{str(row_idx)}/{parse.quote(column)}"
    return dir_path, f"{base_url}/{dir_path}/{filename}"


def create_image_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    image: Image.Image,
    format: str,
    public_assets_storage: PublicAssetsStorage,
) -> ImageSource:
    # get url dir path
    assets_base_url = public_assets_storage.assets_base_url
    overwrite = public_assets_storage.overwrite
    storage_client = public_assets_storage.storage_client

    dir_path, src = generate_asset_src(
        base_url=assets_base_url,
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    object_key = f"{dir_path}/{filename}"
    image_path = f"{storage_client.get_base_directory()}/{object_key}"

    if overwrite or not storage_client.exists(object_key=object_key):
        with storage_client._fs.open(image_path, "wb") as f:
            image.save(fp=f, format=format)
    return ImageSource(src=src, height=image.height, width=image.width)


def create_audio_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    audio_file_bytes: bytes,
    audio_file_extension: str,
    filename: str,
    public_assets_storage: PublicAssetsStorage,
) -> list[AudioSource]:
    # get url dir path
    assets_base_url = public_assets_storage.assets_base_url
    overwrite = public_assets_storage.overwrite
    storage_client = public_assets_storage.storage_client

    dir_path, src = generate_asset_src(
        base_url=assets_base_url,
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    object_key = f"{dir_path}/{filename}"
    audio_path = f"{storage_client.get_base_directory()}/{object_key}"
    suffix = f".{filename.split('.')[-1]}"
    if suffix not in SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE:
        raise ValueError(
            f"Audio format {suffix} is not supported. Supported formats are"
            f" {','.join(SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE)}."
        )
    media_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[suffix]

    if overwrite or not storage_client.exists(object_key=object_key):
        if audio_file_extension == suffix:
            with storage_client._fs.open(audio_path, "wb") as f:
                f.write(audio_file_bytes)
        else:  # we need to convert
            # might spawn a process to convert the audio file using ffmpeg
            with NamedTemporaryFile("wb", suffix=audio_file_extension) as tmpfile:
                tmpfile.write(audio_file_bytes)
                segment: AudioSegment = AudioSegment.from_file(tmpfile.name)
                buffer = BytesIO()
                segment.export(buffer, format=suffix[1:])
                buffer.seek(0)
                with storage_client._fs.open(audio_path, "wb") as f:
                    f.write(buffer.read())
    return [AudioSource(src=src, type=media_type)]
