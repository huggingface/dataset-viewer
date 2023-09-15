# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import os
from collections.abc import Generator
from os import makedirs
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict

from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.storage import StrPath

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"
SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


def create_asset_dir(
    dataset: str, config: str, split: str, row_idx: int, column: str, assets_directory: StrPath
) -> tuple[Path, str]:
    dir_path = Path(assets_directory).resolve() / dataset / DATASET_SEPARATOR / config / split / str(row_idx) / column
    url_dir_path = f"{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{row_idx}/{column}"
    makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    return dir_path, url_dir_path


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


def create_image_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    image: Image.Image,
    assets_base_url: str,
    assets_directory: StrPath,
    overwrite: bool = True,
) -> ImageSource:
    dir_path, url_dir_path = create_asset_dir(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        assets_directory=assets_directory,
    )
    makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    file_path = dir_path / filename
    if overwrite or not file_path.exists():
        image.save(file_path)
    return {
        "src": f"{assets_base_url}/{url_dir_path}/{filename}",
        "height": image.height,
        "width": image.width,
    }


class AudioSource(TypedDict):
    src: str
    type: str


def create_audio_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    audio_file_bytes: bytes,
    audio_file_extension: str,
    assets_base_url: str,
    filename: str,
    assets_directory: StrPath,
    overwrite: bool = True,
) -> list[AudioSource]:
    dir_path, url_dir_path = create_asset_dir(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        assets_directory=assets_directory,
    )
    makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    file_path = dir_path / filename
    if file_path.suffix not in SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE:
        raise ValueError(
            f"Audio format {file_path.suffix} is not supported. Supported formats are"
            f" {','.join(SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE)}."
        )
    media_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[file_path.suffix]
    if overwrite or not file_path.exists():
        if audio_file_extension == file_path.suffix:
            with open(file_path, "wb") as f:
                f.write(audio_file_bytes)
        else:  # we need to convert
            # might spawn a process to convert the audio file using ffmpeg
            with NamedTemporaryFile("wb", suffix=audio_file_extension) as tmpfile:
                tmpfile.write(audio_file_bytes)
                segment: AudioSegment = AudioSegment.from_file(tmpfile.name)
                segment.export(file_path, format=file_path.suffix[1:])
    return [
        {"src": f"{assets_base_url}/{url_dir_path}/{filename}", "type": media_type},
    ]
