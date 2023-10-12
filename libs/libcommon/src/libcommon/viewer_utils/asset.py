# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import logging
import os
from collections.abc import Generator
from os import makedirs
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict, cast
from uuid import uuid4

from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.constants import DATASET_SEPARATOR
from libcommon.storage import StrPath, remove_dir
from libcommon.storage_options import StorageOptions

ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"
SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


def get_and_create_dir_path(assets_directory: StrPath, url_dir_path: str) -> Path:
    dir_path = Path(assets_directory).resolve() / url_dir_path
    makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    return dir_path


def get_url_dir_path(dataset: str, config: str, split: str, row_idx: int, column: str) -> str:
    return f"{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{str(row_idx)}/{column}"


def get_unique_path_for_filename(assets_directory: StrPath, filename: str) -> Path:
    return Path(assets_directory).resolve() / f"{str(uuid4())}-{filename}"


def delete_asset_dir(dataset: str, directory: StrPath) -> None:
    dir_path = Path(directory).resolve() / dataset
    remove_dir(dir_path)


def glob_rows_in_assets_dir(
    dataset: str,
    assets_directory: StrPath,
) -> Generator[Path, None, None]:
    return Path(assets_directory).resolve().glob(os.path.join(dataset, DATASET_SEPARATOR, "*", "*", "*"))


def update_directory_modification_date(path: Path) -> None:
    if path.is_dir():
        # update the directory's last modified date
        temporary_file = path / DATASETS_SERVER_MDATE_FILENAME
        if temporary_file.is_dir():
            raise ValueError(f"Cannot create temporary file {temporary_file} in {path}")
        temporary_file.touch(exist_ok=True)
        if temporary_file.is_file():
            with contextlib.suppress(FileNotFoundError):
                temporary_file.unlink()


def update_last_modified_date_of_rows_in_assets_dir(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    assets_directory: StrPath,
) -> None:
    update_directory_modification_date(Path(assets_directory).resolve() / dataset.split("/")[0])
    row_dirs_path = Path(assets_directory).resolve() / dataset / DATASET_SEPARATOR / config / split
    for row_idx in range(offset, offset + length):
        update_directory_modification_date(row_dirs_path / str(row_idx))


class ImageSource(TypedDict):
    src: str
    height: int
    width: int


class AudioSource(TypedDict):
    src: str
    type: str


def create_image_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    image: Image.Image,
    storage_options: StorageOptions,
    ext: str,
) -> ImageSource:
    # get url dir path
    assets_base_url = storage_options.assets_base_url
    overwrite = storage_options.overwrite
    url_dir_path = get_url_dir_path(dataset=dataset, config=config, split=split, row_idx=row_idx, column=column)
    src = f"{assets_base_url}/{url_dir_path}/{filename}"

    assets_directory = storage_options.assets_directory
    storage_client = storage_options.storage_client
    object_key = f"{assets_directory}/{url_dir_path}/{filename}"

    if overwrite or not storage_client.exists(object_key=object_key):
        image_path = f"{storage_client._storage_root}/{object_key}"
        with storage_client._fs.open(image_path, 'wb') as f:
            image.save(fp=f, format="JPEG")
    return ImageSource(src=src, height=image.height, width=image.width)



def create_audio_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    audio_file_bytes: bytes,
    audio_file_extension: str,
    filename: str,
    storage_options: StorageOptions,
) -> list[AudioSource]:
    # get url dir path
    assets_base_url = storage_options.assets_base_url
    overwrite = storage_options.overwrite
    url_dir_path = get_url_dir_path(dataset=dataset, config=config, split=split, row_idx=row_idx, column=column)
    src = f"{assets_base_url}/{url_dir_path}/{filename}"

    assets_directory = storage_options.assets_directory
    storage_client = storage_options.storage_client
    suffix = f".{filename.split('.')[-1]}"
    if suffix not in SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE:
            raise ValueError(
                f"Audio format {suffix} is not supported. Supported formats are"
                f" {','.join(SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE)}."
            )
    media_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[suffix]
    object_key = f"{assets_directory}/{url_dir_path}/{filename}"
    audio_path = f"{storage_client._storage_root}/{object_key}"
    if overwrite or not storage_client.exists(object_key=object_key):
        if audio_file_extension == suffix:
            with storage_client._fs.open(audio_path, 'wb') as f:
                f.write(audio_file_bytes)
        else:  # we need to convert
                # might spawn a process to convert the audio file using ffmpeg
            print("convertion firts")
            with NamedTemporaryFile("wb", suffix=audio_file_extension) as tmpfile:
                tmpfile.write(audio_file_bytes)
                segment: AudioSegment = AudioSegment.from_file(tmpfile.name)
                print("convertion temp done")
                with storage_client._fs.open(audio_path, 'wb') as f:
                    segment.export(f, format=suffix[1:])
                    print("segment export done")
    return AudioSource(src=src, type=media_type)
