# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import logging
import os
from collections.abc import Callable, Generator
from functools import partial
from os import makedirs
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, TypedDict, Union, cast
from uuid import uuid4

from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.constants import DATASET_SEPARATOR
from libcommon.s3_client import S3Client
from libcommon.storage import StrPath, remove_dir
from libcommon.storage_options import DirectoryStorageOptions, S3StorageOptions

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


SupportedSource = Union[ImageSource, AudioSource]


def upload_asset_file(
    url_dir_path: str,
    filename: str,
    file_path: Path,
    overwrite: bool = True,
    s3_client: Optional[S3Client] = None,
    s3_folder_name: Optional[str] = None,
) -> None:
    if s3_client is not None:
        object_key = f"{s3_folder_name}/{url_dir_path}/{filename}"
        if overwrite or not s3_client.exists(object_key):
            s3_client.upload(str(file_path), object_key)
            logging.debug(f"{object_key=} has been uploaded")
            os.remove(file_path)


def create_asset_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    storage_options: Union[DirectoryStorageOptions, S3StorageOptions],
    fn: Callable[[Path, str, bool], SupportedSource],
) -> SupportedSource:
    # get url dir path
    assets_base_url = storage_options.assets_base_url
    assets_directory = storage_options.assets_directory
    overwrite = storage_options.overwrite
    use_s3_storage = isinstance(storage_options, S3StorageOptions)
    logging.debug(f"storage options with {use_s3_storage=}")
    url_dir_path = get_url_dir_path(dataset=dataset, config=config, split=split, row_idx=row_idx, column=column)
    src = f"{assets_base_url}/{url_dir_path}/{filename}"

    # configure file path
    file_path = (
        get_unique_path_for_filename(assets_directory, filename)
        if use_s3_storage
        else get_and_create_dir_path(
            assets_directory=assets_directory,
            url_dir_path=url_dir_path,
        )
        / filename
    )

    # create file locally
    asset_file = fn(file_path=file_path, src=src, overwrite=overwrite)  # type: ignore

    # upload to s3 if enabled
    if use_s3_storage:
        s3_storage_options: S3StorageOptions = cast(S3StorageOptions, storage_options)
        s3_folder_name = s3_storage_options.s3_folder_name
        s3_client = s3_storage_options.s3_client

        upload_asset_file(
            url_dir_path=url_dir_path,
            filename=filename,
            file_path=file_path,
            overwrite=overwrite,
            s3_client=s3_client,
            s3_folder_name=s3_folder_name,
        )

    return asset_file


def save_image(image: Image.Image, file_path: Path, src: str, overwrite: bool) -> ImageSource:
    if overwrite or not file_path.exists():
        image.save(file_path)
    return ImageSource(src=src, height=image.height, width=image.width)


def save_audio(
    audio_file_bytes: bytes, audio_file_extension: str, file_path: Path, src: str, overwrite: bool
) -> AudioSource:
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

    return AudioSource(src=src, type=media_type)


def create_image_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    image: Image.Image,
    storage_options: DirectoryStorageOptions,
) -> ImageSource:
    fn = partial(save_image, image=image)
    return cast(
        ImageSource,
        create_asset_file(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=row_idx,
            column=column,
            filename=filename,
            storage_options=storage_options,
            fn=fn,
        ),
    )


def create_audio_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    audio_file_bytes: bytes,
    audio_file_extension: str,
    filename: str,
    storage_options: DirectoryStorageOptions,
) -> list[AudioSource]:
    fn = partial(save_audio, audio_file_bytes=audio_file_bytes, audio_file_extension=audio_file_extension)
    return [
        cast(
            AudioSource,
            create_asset_file(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                column=column,
                filename=filename,
                storage_options=storage_options,
                fn=fn,
            ),
        )
    ]
