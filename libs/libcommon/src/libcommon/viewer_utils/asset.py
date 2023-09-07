# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import json
import os
from functools import partial
from hashlib import sha1
from os import makedirs
from pathlib import Path
from typing import Generator, List, Optional, Tuple, TypedDict, Union
from uuid import uuid4
from tempfile import NamedTemporaryFile

from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.s3_client import S3Client
from libcommon.storage import StrPath

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"
S3_RESOURCE = "s3"
SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


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


class AudioSource(TypedDict):
    src: str
    type: str


SupportedSource = Union[ImageSource, AudioSource]


def create_asset_file(
    dataset: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    assets_base_url: str,
    assets_directory: StrPath,
    fn: partial,  # type: ignore
    overwrite: bool = True,
    # TODO: Once assets and cached-assets are migrated to S3, this parameter is no more needed
    use_s3_storage: bool = False,
    s3_client: Optional[S3Client] = None,
    # TODO: Once assets and cached-assets are migrated to S3, the following parameters dont need to be optional
    s3_bucket: Optional[str] = None,
    s3_folder_name: Optional[str] = None,
) -> SupportedSource:
    dir_path, url_dir_path = get_asset_dir_name(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        assets_directory=assets_directory,
    )

    if not use_s3_storage:
        file_path = dir_path / filename
        makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    else:
        payload = (
            str(uuid4()),
            dataset,
            config,
            split,
            row_idx,
            column,
        )
        prefix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
        file_path = Path(assets_directory).resolve() / f"{prefix}-{filename}"

    src = f"{assets_base_url}/{url_dir_path}/{filename}"
    source = fn(file_path=file_path, src=src, overwrite=overwrite)

    if use_s3_storage and s3_client is not None and s3_bucket is not None:
        object_key = f"{s3_folder_name}/{url_dir_path}/{filename}"
        create_object = overwrite or not s3_client.exists_in_bucket(s3_bucket, object_key)
        if create_object:
            s3_client.upload_to_bucket(str(file_path), s3_bucket, object_key)
            os.remove(file_path)

    return source


def save_image(image: Image.Image, file_path: str, src: str, overwrite: bool) -> ImageSource:
    if overwrite or not file_path.exists():
        image.save(file_path)
    return ImageSource(src=src, height=image.height, width=image.width)


def save_audio(
    audio_file_bytes: bytes, audio_file_extension: str, file_path: str, src: str, overwrite: bool
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
    assets_base_url: str,
    assets_directory: StrPath,
    overwrite: bool = True,
    # TODO: Once assets and cached-assets are migrated to S3, this parameter is no more needed
    use_s3_storage: bool = False,
    s3_client: Optional[S3Client] = None,
    # TODO: Once assets and cached-assets are migrated to S3, the following parameters dont need to be optional
    s3_bucket: Optional[str] = None,
    s3_folder_name: Optional[str] = None,
) -> ImageSource:
    fn = partial(save_image, image=image)
    return create_asset_file(
        dataset=dataset,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
        assets_base_url=assets_base_url,
        assets_directory=assets_directory,
        fn=fn,
        overwrite=overwrite,
        use_s3_storage=use_s3_storage,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
        s3_folder_name=s3_folder_name,
    )


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
    # TODO: Once assets and cached-assets are migrated to S3, this parameter is no more needed
    use_s3_storage: bool = False,
    s3_client: Optional[S3Client] = None,
    # TODO: Once assets and cached-assets are migrated to S3, the following parameters dont need to be optional
    s3_bucket: Optional[str] = None,
    s3_folder_name: Optional[str] = None,
) -> List[AudioSource]:
    fn = partial(save_audio, audio_file_bytes=audio_file_bytes, audio_file_extension=audio_file_extension)
    return [
        create_asset_file(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=row_idx,
            column=column,
            filename=filename,
            assets_base_url=assets_base_url,
            assets_directory=assets_directory,
            fn=fn,
            overwrite=overwrite,
            use_s3_storage=use_s3_storage,
            s3_client=s3_client,
            s3_bucket=s3_bucket,
            s3_folder_name=s3_folder_name,
        )
    ]
