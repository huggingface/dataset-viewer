# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from os import makedirs
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import soundfile  # type:ignore
from numpy import ndarray  # type:ignore
from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755


def create_asset_dir(
    dataset: str, config: str, split: str, row_idx: int, column: str, assets_directory: str
) -> Tuple[Path, str]:
    dir_path = Path(assets_directory).resolve() / dataset / DATASET_SEPARATOR / config / split / str(row_idx) / column
    url_dir_path = f"{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{row_idx}/{column}"
    makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    return dir_path, url_dir_path


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
    assets_directory: Optional[str],
) -> ImageSource:
    dir_path, url_dir_path = create_asset_dir(
        dataset=dataset, config=config, split=split, row_idx=row_idx, column=column, assets_directory=assets_directory
    )
    file_path = dir_path / filename
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
    array: ndarray,
    sampling_rate: int,
    assets_base_url: str,
    filename_base: str,
    assets_directory: Optional[str],
) -> List[AudioSource]:
    wav_filename = f"{filename_base}.wav"
    mp3_filename = f"{filename_base}.mp3"
    dir_path, url_dir_path = create_asset_dir(
        dataset=dataset, config=config, split=split, row_idx=row_idx, column=column, assets_directory=assets_directory
    )
    wav_file_path = dir_path / wav_filename
    mp3_file_path = dir_path / mp3_filename
    soundfile.write(wav_file_path, array, sampling_rate)
    segment = AudioSegment.from_wav(wav_file_path)
    segment.export(mp3_file_path, format="mp3")
    return [
        {"src": f"{assets_base_url}/{url_dir_path}/{mp3_filename}", "type": "audio/mpeg"},
        {"src": f"{assets_base_url}/{url_dir_path}/{wav_filename}", "type": "audio/wav"},
    ]
