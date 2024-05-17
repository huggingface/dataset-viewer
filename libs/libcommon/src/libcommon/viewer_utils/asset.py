# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, TypedDict
from urllib import parse

from PIL import Image, ImageOps
from pydub import AudioSegment  # type:ignore

from libcommon.constants import DATASET_SEPARATOR
from libcommon.storage import StrPath, remove_dir
from libcommon.storage_client import StorageClient

SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".opus": "audio/opus"}


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


def generate_object_key(
    dataset: str, revision: str, config: str, split: str, row_idx: int, column: str, filename: str
) -> str:
    return f"{parse.quote(dataset)}/{DATASET_SEPARATOR}/{revision}/{DATASET_SEPARATOR}/{parse.quote(config)}/{parse.quote(split)}/{str(row_idx)}/{parse.quote(column)}/{filename}"


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
    storage_client: StorageClient,
) -> ImageSource:
    object_key = generate_object_key(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    if storage_client.overwrite or not storage_client.exists(object_key):
        image = ImageOps.exif_transpose(image)  # type: ignore[assignment]
        buffer = BytesIO()
        image.save(fp=buffer, format=format)
        buffer.seek(0)
        with storage_client._fs.open(storage_client.get_full_path(object_key), "wb") as f:
            f.write(buffer.read())
    return ImageSource(src=storage_client.get_url(object_key), height=image.height, width=image.width)


def create_audio_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    audio_file_bytes: bytes,
    audio_file_extension: Optional[str],
    filename: str,
    storage_client: StorageClient,
) -> list[AudioSource]:
    object_key = generate_object_key(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    suffix = f".{filename.split('.')[-1]}"
    if suffix not in SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE:
        raise ValueError(
            f"Audio format {suffix} is not supported. Supported formats are"
            f" {','.join(SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE)}."
        )
    media_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[suffix]

    if storage_client.overwrite or not storage_client.exists(object_key):
        audio_path = storage_client.get_full_path(object_key)
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
    return [AudioSource(src=storage_client.get_url(object_key), type=media_type)]
