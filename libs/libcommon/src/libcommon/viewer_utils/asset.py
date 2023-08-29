# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import io
import json
import os
import random
from hashlib import sha1
from os import makedirs
from pathlib import Path
from typing import Generator, List, Optional, Tuple, TypedDict

import boto3
import soundfile  # type:ignore
from numpy import ndarray
from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.storage import StrPath

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"
S3_RESOURCE = "s3"


def get_asset_dir_name(
    dataset: str, config: str, split: str, row_idx: int, column: str, assets_directory: StrPath
) -> Tuple[Path, str]:
    dir_path = Path(assets_directory).resolve() / dataset / DATASET_SEPARATOR / config / split / str(row_idx) / column
    url_dir_path = f"{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{row_idx}/{column}"
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
    use_s3_storage: bool = False,
    s3_bucket: Optional[str] = None,
    s3_access_key_id: Optional[str] = None,
    s3_secret_access_key: Optional[str] = None,
    s3_region: Optional[str] = None,
    s3_folder_name: Optional[str] = None,
) -> ImageSource:
    dir_path, url_dir_path = get_asset_dir_name(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        assets_directory=assets_directory,
    )
    if use_s3_storage:
        key = f"{s3_folder_name}/{url_dir_path}/{filename}"
        s3_client = boto3.client(
            S3_RESOURCE,
            region_name=s3_region,
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
        )
        exists = True
        if not overwrite:
            try:
                s3_client.head_object(
                    Bucket=s3_bucket,
                    Key=key,
                )
            except Exception:
                exists = False
        if overwrite or not exists:
            image_byte_arr = io.BytesIO()
            image.save(image_byte_arr, format="JPEG")
            image_byte_arr.seek(0)
            s3_client.upload_fileobj(image_byte_arr, s3_bucket, key)
    else:
        makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
        file_path = dir_path / filename
        if overwrite or not file_path.exists():
            image.save(file_path)
    src = f"{assets_base_url}/{url_dir_path}/{filename}"
    return {
        "src": src,
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
    use_s3_storage: bool = False,
    s3_bucket: Optional[str] = None,
    s3_access_key_id: Optional[str] = None,
    s3_secret_access_key: Optional[str] = None,
    s3_region: Optional[str] = None,
    s3_folder_name: Optional[str] = None,
) -> List[AudioSource]:
    wav_filename = f"{filename_base}.wav"
    mp3_filename = f"{filename_base}.mp3"
    dir_path, url_dir_path = get_asset_dir_name(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        assets_directory=assets_directory,
    )
    if not use_s3_storage:
        wav_file_path = dir_path / wav_filename
        mp3_file_path = dir_path / mp3_filename
        makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    else:
        random_str = f"{random.randrange(10**13, 10**14)}"  # nosec B311
        payload = (
            random_str,
            dataset,
            config,
            split,
            row_idx,
            column,
        )
        prefix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
        wav_file_path = Path(assets_directory).resolve() / f"{prefix}-{wav_filename}"
        mp3_file_path = Path(assets_directory).resolve() / f"{prefix}-{mp3_filename}"

    if overwrite or not wav_file_path.exists():
        soundfile.write(wav_file_path, array, sampling_rate)
    if overwrite or not mp3_file_path.exists():
        segment = AudioSegment.from_wav(wav_file_path)
        segment.export(mp3_file_path, format="mp3")

    if use_s3_storage:
        wav_key = f"{s3_folder_name}/{url_dir_path}/{wav_filename}"
        mp3_key = f"{s3_folder_name}/{url_dir_path}/{mp3_filename}"

        s3_client = boto3.client(
            S3_RESOURCE,
            region_name=s3_region,
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
        )
        wav_exists = True
        mp3_exists = True
        if not overwrite:
            try:
                s3_client.head_object(
                    Bucket=s3_bucket,
                    Key=wav_key,
                )
            except Exception:
                wav_exists = False
            try:
                s3_client.head_object(
                    Bucket=s3_bucket,
                    Key=mp3_key,
                )
            except Exception:
                mp3_exists = False

        if overwrite or not wav_exists:
            s3_client.upload_file(wav_file_path, s3_bucket, wav_key)
            os.remove(wav_file_path)
        if overwrite or not mp3_exists:
            s3_client.upload_file(mp3_file_path, s3_bucket, mp3_key)
            os.remove(mp3_file_path)

    return [
        {"src": f"{assets_base_url}/{url_dir_path}/{mp3_filename}", "type": "audio/mpeg"},
        {"src": f"{assets_base_url}/{url_dir_path}/{wav_filename}", "type": "audio/wav"},
    ]
