# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Optional, TypedDict

from PIL import Image, ImageOps
from pydub import AudioSegment  # type:ignore

if TYPE_CHECKING:
    from libcommon.storage_client import StorageClient


SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".opus": "audio/ogg"}
SUPPORTED_AUDIO_EXTENSIONS = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE.keys()
DATASET_GIT_REVISION_PLACEHOLDER = "{dataset_git_revision}"


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
    format: str,
    storage_client: "StorageClient",
) -> ImageSource:
    # We use a placeholder revision in the JSON stored in the database,
    # while the path of the file stored on the disk/s3 contains the revision.
    # The placeholder will be replaced later by the
    # dataset_git_revision of cache responses when the data will be accessed.
    # This is useful to allow moving files to a newer revision without having
    # to modify the cached rows content.
    object_key = storage_client.generate_object_key(
        dataset=dataset,
        revision=DATASET_GIT_REVISION_PLACEHOLDER,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    path = replace_dataset_git_revision_placeholder(object_key, revision=revision)
    if storage_client.overwrite or not storage_client.exists(path):
        image = ImageOps.exif_transpose(image)  # type: ignore[assignment]
        buffer = BytesIO()
        image.save(fp=buffer, format=format)
        buffer.seek(0)
        with storage_client._fs.open(storage_client.get_full_path(path), "wb") as f:
            f.write(buffer.read())
    return ImageSource(
        src=storage_client.get_url(object_key, revision=revision), height=image.height, width=image.width
    )


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
    storage_client: "StorageClient",
) -> list[AudioSource]:
    # We use a placeholder revision that will be replaced later by the
    # dataset_git_revision of cache responses when the data will be accessed.
    # This is useful to allow moving files to a newer revision without having
    # to modify the cached rows content.
    object_key = storage_client.generate_object_key(
        dataset=dataset,
        revision=DATASET_GIT_REVISION_PLACEHOLDER,
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

    path = replace_dataset_git_revision_placeholder(object_key, revision=revision)
    if storage_client.overwrite or not storage_client.exists(path):
        audio_path = storage_client.get_full_path(path)
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
    return [AudioSource(src=storage_client.get_url(object_key, revision=revision), type=media_type)]


def replace_dataset_git_revision_placeholder(url_or_object_key: str, revision: str) -> str:
    # Set the right revision in the URL e.g.
    # Before: https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/{dataset_git_revision}/--/default/test/0/image/image.jpg
    # After:  https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/c844916c2920d2d01e8a15f8dc1caf6f017a293c/--/default/test/0/image/image.jpg
    return url_or_object_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, revision)
