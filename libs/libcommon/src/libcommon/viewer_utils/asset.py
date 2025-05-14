# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from io import BufferedReader, BytesIO
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

from pdfplumber.pdf import PDF
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


class VideoSource(TypedDict):
    src: str


class PDFSource(TypedDict):
    src: str
    thumbnail_src: str
    thumbnail_height: int
    thumbnail_width: int


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
    object_path = storage_client.generate_object_path(
        dataset=dataset,
        revision=DATASET_GIT_REVISION_PLACEHOLDER,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    path = replace_dataset_git_revision_placeholder(object_path, revision=revision)
    if storage_client.overwrite or not storage_client.exists(path):
        image = ImageOps.exif_transpose(image)  # type: ignore[assignment]
        buffer = BytesIO()
        image.save(fp=buffer, format=format)
        buffer.seek(0)
        with storage_client._fs.open(storage_client.get_full_path(path), "wb") as f:
            f.write(buffer.read())
    return ImageSource(
        src=storage_client.get_url(object_path, revision=revision), height=image.height, width=image.width
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
    object_path = storage_client.generate_object_path(
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

    path = replace_dataset_git_revision_placeholder(object_path, revision=revision)
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
    return [AudioSource(src=storage_client.get_url(object_path, revision=revision), type=media_type)]


def create_pdf_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    pdf: PDF,
    storage_client: "StorageClient",
) -> PDFSource:
    thumbnail_object_path = storage_client.generate_object_path(
        dataset=dataset,
        revision=DATASET_GIT_REVISION_PLACEHOLDER,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=f"{filename}.png",
    )
    thumbnail_storage_path = replace_dataset_git_revision_placeholder(thumbnail_object_path, revision=revision)
    thumbnail = pdf.pages[0].to_image()

    if storage_client.overwrite or not storage_client.exists(thumbnail_storage_path):
        thumbnail_buffer = BytesIO()
        thumbnail.save(thumbnail_buffer)
        thumbnail_buffer.seek(0)
        with storage_client._fs.open(storage_client.get_full_path(thumbnail_storage_path), "wb") as thumbnail_file:
            thumbnail_file.write(thumbnail_buffer.read())

    pdf_object_path = storage_client.generate_object_path(
        dataset=dataset,
        revision=DATASET_GIT_REVISION_PLACEHOLDER,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    pdf_storage_path = replace_dataset_git_revision_placeholder(pdf_object_path, revision=revision)

    def is_valid_pdf(pdf_stream: Union[BufferedReader, BytesIO]) -> bool:
        current_position = pdf_stream.tell()
        try:
            pdf_stream.seek(0)
            return pdf_stream.read(5) == b"%PDF-"
        finally:
            pdf_stream.seek(current_position)

    pdf_data = pdf.stream
    if not is_valid_pdf(pdf_data):
        raise ValueError("The provided data is not a valid PDF.")

    if storage_client.overwrite or not storage_client.exists(pdf_storage_path):
        with storage_client._fs.open(storage_client.get_full_path(pdf_storage_path), "wb") as pdf_file:
            pdf_data.seek(0)
            pdf_file.write(pdf_data.read())

    return PDFSource(
        src=storage_client.get_url(pdf_object_path, revision=revision),
        thumbnail_src=storage_client.get_url(thumbnail_object_path, revision=revision),
        thumbnail_height=thumbnail.annotated.height,
        thumbnail_width=thumbnail.annotated.width,
    )


def replace_dataset_git_revision_placeholder(url_or_object_path: str, revision: str) -> str:
    # Set the right revision in the URL e.g.
    # Before: https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/{dataset_git_revision}/--/default/test/0/image/image.jpg
    # After:  https://datasets-server.huggingface.co/assets/vidore/syntheticDocQA_artificial_intelligence_test/--/c844916c2920d2d01e8a15f8dc1caf6f017a293c/--/default/test/0/image/image.jpg
    return url_or_object_path.replace(DATASET_GIT_REVISION_PLACEHOLDER, revision)


def create_video_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    encoded_video: dict[str, Any],
    storage_client: "StorageClient",
) -> VideoSource:
    # We use a placeholder revision in the JSON stored in the database,
    # while the path of the file stored on the disk/s3 contains the revision.
    # The placeholder will be replaced later by the
    # dataset_git_revision of cache responses when the data will be accessed.
    # This is useful to allow moving files to a newer revision without having
    # to modify the cached rows content.
    if "path" in encoded_video and isinstance(encoded_video["path"], str) and "://" in encoded_video["path"]:
        # in general video files are stored in the dataset repository, we can just get the URL
        # (`datasets` doesn't embed the video bytes in Parquet when the file is already on HF)
        object_path = encoded_video["path"].replace(revision, DATASET_GIT_REVISION_PLACEHOLDER)
    elif "bytes" in encoded_video and isinstance(encoded_video["bytes"], bytes):
        # (rare and not very important) otherwise we attempt to upload video data from webdataset/parquet files but don't process them
        object_path = storage_client.generate_object_path(
            dataset=dataset,
            revision=DATASET_GIT_REVISION_PLACEHOLDER,
            config=config,
            split=split,
            row_idx=row_idx,
            column=column,
            filename=filename,
        )
        path = replace_dataset_git_revision_placeholder(object_path, revision=revision)
        if storage_client.overwrite or not storage_client.exists(path):
            with storage_client._fs.open(storage_client.get_full_path(path), "wb") as f:
                f.write(encoded_video["bytes"])
    else:
        raise ValueError("The video cell doesn't contain a valid path or bytes")
    src = storage_client.get_url(object_path, revision=revision)
    return VideoSource(src=src)
