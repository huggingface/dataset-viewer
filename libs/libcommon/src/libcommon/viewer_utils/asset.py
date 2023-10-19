# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict

from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

from libcommon.constants import DATASET_SEPARATOR
from libcommon.storage import StrPath, remove_dir
from libcommon.storage_options import StorageOptions

ASSET_DIR_MODE = 0o755
DATASETS_SERVER_MDATE_FILENAME = ".dss"
SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


def get_url_dir_path(dataset: str, revision: str, config: str, split: str, row_idx: int, column: str) -> str:
    return f"{dataset}/{DATASET_SEPARATOR}/{revision}/{DATASET_SEPARATOR}/{config}/{split}/{str(row_idx)}/{column}"


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


def create_image_file(
    dataset: str,
    revision: str,
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
    url_dir_path = get_url_dir_path(revision=revision, dataset=dataset, config=config, split=split, row_idx=row_idx, column=column)
    src = f"{assets_base_url}/{url_dir_path}/{filename}"

    storage_client = storage_options.storage_client
    object_key = f"{storage_client._folder}/{url_dir_path}/{filename}"
    image_path = f"{storage_client._storage_root}/{object_key}"

    if overwrite or not storage_client.exists(object_key=object_key):
        print(f"About to save in {image_path=}")
        with storage_client._fs.open(image_path, 'wb') as f:
            image.save(fp=f, format="JPEG")
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
    storage_options: StorageOptions,
) -> list[AudioSource]:
    # get url dir path
    assets_base_url = storage_options.assets_base_url
    overwrite = storage_options.overwrite
    url_dir_path = get_url_dir_path(revision=revision, dataset=dataset, config=config, split=split, row_idx=row_idx, column=column)
    src = f"{assets_base_url}/{url_dir_path}/{filename}"

    storage_client = storage_options.storage_client
    suffix = f".{filename.split('.')[-1]}"
    if suffix not in SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE:
            raise ValueError(
                f"Audio format {suffix} is not supported. Supported formats are"
                f" {','.join(SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE)}."
            )
    media_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[suffix]
    object_key = f"{storage_client._folder}/{url_dir_path}/{filename}"
    audio_path = f"{storage_client._storage_root}/{object_key}"
    if overwrite or not storage_client.exists(object_key=object_key):
        if audio_file_extension == suffix:
            print(f"About to save in {audio_path=}")
            with storage_client._fs.open(audio_path, 'wb') as f:
                f.write(audio_file_bytes)
        else:  # we need to convert
                # might spawn a process to convert the audio file using ffmpeg
            print("convertion firts")
            with NamedTemporaryFile("wb", suffix=audio_file_extension) as tmpfile:
                tmpfile.write(audio_file_bytes)
                segment: AudioSegment = AudioSegment.from_file(tmpfile.name)
                print("convertion temp done")
                print(f"About to save converted to {audio_path=}")
                with storage_client._fs.open(audio_path, 'wb') as f:
                    segment.export(f, format=suffix[1:])
                    print("segment export done")
    return AudioSource(src=src, type=media_type)
