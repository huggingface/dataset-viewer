# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import os
from os import makedirs
from pathlib import Path
from typing import Generator, List, Tuple, TypedDict

import soundfile  # type:ignore
from numpy import ndarray
from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.storage import StrPath

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"


def create_asset_dir(
    dataset: str, config: str, split: str, row_idx: int, column: str, assets_directory: StrPath
) -> Tuple[Path, str]:
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


def create_audio_files(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    array: ndarray,  # type: ignore
    sampling_rate: int,
    assets_base_url: str,
    filename_base: str,
    assets_directory: StrPath,
    overwrite: bool = True,
) -> List[AudioSource]:
    wav_filename = f"{filename_base}.wav"
    mp3_filename = f"{filename_base}.mp3"
    dir_path, url_dir_path = create_asset_dir(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        assets_directory=assets_directory,
    )
    makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    wav_file_path = dir_path / wav_filename
    mp3_file_path = dir_path / mp3_filename
    if overwrite or not wav_file_path.exists():
        soundfile.write(wav_file_path, array, sampling_rate)
    if overwrite or not mp3_file_path.exists():
        segment = AudioSegment.from_wav(wav_file_path)
        segment.export(mp3_file_path, format="mp3")
    return [
        {"src": f"{assets_base_url}/{url_dir_path}/{mp3_filename}", "type": "audio/mpeg"},
        {"src": f"{assets_base_url}/{url_dir_path}/{wav_filename}", "type": "audio/wav"},
    ]
