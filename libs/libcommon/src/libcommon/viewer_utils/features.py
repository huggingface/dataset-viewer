# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import logging
import os
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional, Union
from zlib import adler32

import datasets.config
import numpy as np
import pdfplumber
from datasets import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    ClassLabel,
    Features,
    Image,
    Json,
    LargeList,
    List,
    Mesh,
    Pdf,
    Translation,
    TranslationVariableLanguages,
    Value,
    Video,
)
from datasets.utils.file_utils import is_local_path
from huggingface_hub import HfFileSystem

from libcommon.dtos import FeatureItem
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import hf_to_https_url
from libcommon.viewer_utils.asset import (
    MESH_THUMBNAIL_MAX_POINTS,
    SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE,
    SUPPORTED_AUDIO_EXTENSIONS,
    AudioSource,
    ImageSource,
    MeshSource,
    PDFSource,
    VideoSource,
    create_audio_file,
    create_image_file,
    create_mesh_file,
    create_mesh_thumbnail,
    create_pdf_file,
    create_video_file,
)

if TYPE_CHECKING:
    import trimesh


AUDIO_FILE_MAGIC_NUMBERS: dict[str, Any] = {
    ".wav": [
        [(b"\x52\x49\x46\x46", 0), (b"\x57\x41\x56\x45", 8)],
        [(b"RIFF", 0), (b"WAVE", 8)],
    ],  # AND: (magic_number, start)
    ".mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\x49\x44\x33", b"ID3"],  # OR
    ".flac": [b"fLaC"],
}

VIDEO_FILE_MAGIC_NUMBERS: dict[str, Any] = {
    ".mkv": [b"\x1aE\xdf\xa3"],
    ".mp4": [b"ftypisom", (b"ftypisom", 4), b"ftypMSNV", (b"ftypMSNV", 4)],
    ".avi": [[(b"RIFF", 0), (b"AVI ", 8)]],
    ".mpeg": [b"\x00\x00\x01\xba"],
    ".mov": [b"\x00\x00\x01\xb3"],
}

# the mesh file types that the viewer renders: trimesh reads them without additional dependency, except "sog"
SUPPORTED_MESH_FILE_TYPES = {"glb", "ply", "stl", "obj", "sog"}
SUPPORTED_MESH_EXTENSIONS = ", ".join(f".{file_type}" for file_type in sorted(SUPPORTED_MESH_FILE_TYPES))
# a SOG file stores one splat per pixel of each of its images: 8192x8192 pixels is 67M splats, more than any
# scene we know of, and it bounds the memory of a file whose images are much larger than the scene they hold
SOG_MAX_PIXELS = 8192 * 8192
# the images of a SOG file are about the size of its scene: one pixel per splat, plus the padding of the image
# sides (to a multiple of 4 for PlayCanvas, or to a power of 2 for other writers, up to 4 pixels per splat)
SOG_MAX_PIXELS_PER_SPLAT = 4
SOG_MAX_PADDING = 64 * 64
# the largest decompressed size of a file of a SOG archive: its WebP images are already compressed
SOG_MAX_MEMBER_BYTES = 512 * 1024 * 1024
MESH_FILE_TYPE_ERROR = (
    "Decoding mesh bytes requires a 'path' value with one of the supported mesh extensions "
    f"({SUPPORTED_MESH_EXTENSIONS}) to infer the mesh file type."
)
# rendering the thumbnail of a mesh file takes about 9 times its size in memory (a 618 MB glb file takes 5.5 GiB),
# and a rows pod has 12 GiB for all its requests: the larger files are not rendered, and their cell is empty
MESH_MAX_FILE_SIZE = 300_000_000


logging.getLogger("pdfminer").setLevel(logging.ERROR)


def append_hash_suffix(string: str, json_path: Optional[list[Union[str, int]]] = None) -> str:
    """
    Hash the json path to a string.
    Details:
    - no suffix if the list is empty
    - converted to hexadecimal to make the hash shorter
    - the 0x prefix is removed

    Args:
        string (`str`): The string to append the hash to.
        json_path (`list(str|int)`): the json path, which is a list of keys and indices
    Returns:
        `str`: the string suffixed with the hash of the json path
    """
    return f"{string}-{hex(adler32(json.dumps(json_path).encode()))[2:]}" if json_path else string


def image(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    storage_client: StorageClient,
    hf_endpoint: str,
    hf_token: Optional[str],
    json_path: Optional[list[Union[str, int]]] = None,
) -> Optional[ImageSource]:
    from PIL import Image as PILImage

    if value is None:
        return None
    if isinstance(value, dict) and value.get("bytes"):
        value = PILImage.open(BytesIO(value["bytes"]))
    elif isinstance(value, bytes):
        value = PILImage.open(BytesIO(value))
    elif isinstance(value, dict) and "path" in value and isinstance(value["path"], str):
        if is_local_path(value["path"]):
            return None
        elif value["path"].startswith(f"hf://datasets/{dataset}@"):
            with HfFileSystem(endpoint=hf_endpoint, token=hf_token).open(value["path"], "rb") as f:
                src = hf_to_https_url(value["path"], hf_endpoint)
                image = PILImage.open(f)
                return ImageSource(src=src, height=image.height, width=image.width)

    if not isinstance(value, PILImage.Image):
        raise TypeError(
            "Image cell must be a PIL image or an encoded dict of an image, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )
    # attempt to generate one of the supported formats; if unsuccessful, throw an error
    for ext, format in [(".jpg", "JPEG"), (".png", "PNG")]:
        try:
            return create_image_file(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                column=featureName,
                filename=f"{append_hash_suffix('image', json_path)}{ext}",
                image=value,
                format=format,
                storage_client=storage_client,
            )
        except OSError:
            # if wrong format, try the next one, see https://github.com/huggingface/dataset-viewer/issues/191
            #  OSError: cannot write mode P as JPEG
            #  OSError: cannot write mode RGBA as JPEG
            continue
    raise ValueError("Image cannot be written as JPEG or PNG")


def audio(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    storage_client: StorageClient,
    hf_endpoint: str,
    hf_token: Optional[str],
    json_path: Optional[list[Union[str, int]]] = None,
) -> Optional[list[AudioSource]]:
    from datasets.features._torchcodec import AudioDecoder

    if value is None:
        return None
    if isinstance(value, AudioDecoder) and hasattr(value, "_hf_encoded") and isinstance(value._hf_encoded, dict):
        value = value._hf_encoded  # `datasets` patches `torchcodec` to store the encoded data here
    elif isinstance(value, dict):
        value = {"path": value.get("path"), "bytes": value["bytes"]}
    elif isinstance(value, bytes):
        value = {"path": None, "bytes": value}
    elif isinstance(value, str):
        value = {"path": value, "bytes": None}

    if not isinstance(value, dict):
        raise TypeError(
            "Audio cell must be an encoded dict of an audio sample, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )

    audio_file_extension = get_audio_file_extension(value)
    if "path" in value and isinstance(value["path"], str) and value.get("bytes") is None:
        if audio_file_extension in SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE:
            if value["path"].startswith(f"hf://datasets/{dataset}@"):
                src = hf_to_https_url(value["path"], hf_endpoint)
                return [AudioSource(src=src, type=SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[audio_file_extension])]
        if is_local_path(value["path"]):
            return None
        elif value["path"].startswith(f"hf://datasets/{dataset}@"):
            with HfFileSystem(endpoint=hf_endpoint, token=hf_token).open(value["path"], "rb") as f:
                audio_file_bytes = f.read()
        else:
            raise ValueError(f"invalid path: {value['path']}")
    elif "bytes" in value and isinstance(value["bytes"], bytes):
        audio_file_bytes = value["bytes"]
    else:
        raise ValueError(
            "Audio cell dict must be an audio sample, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )

    if not audio_file_extension:
        audio_file_extension = infer_audio_file_extension(audio_file_bytes)
    # convert to wav if the audio file extension is not supported
    target_audio_file_extension = (
        audio_file_extension if audio_file_extension in SUPPORTED_AUDIO_EXTENSIONS else ".wav"
    )
    # this function can raise, we don't catch it
    return create_audio_file(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=featureName,
        audio_file_bytes=audio_file_bytes,
        audio_file_extension=audio_file_extension,
        storage_client=storage_client,
        filename=f"{append_hash_suffix('audio', json_path)}{target_audio_file_extension}",
    )


def get_audio_file_extension(value: Any) -> Optional[str]:
    from datasets.features._torchcodec import AudioDecoder

    if isinstance(value, dict) and "path" in value:
        if isinstance(value["path"], str):
            # .split("::")[0] for chained URLs like zip://audio.wav::https://foo.bar/data.zip
            # It might be "" for audio files downloaded from the Hub: make it None
            audio_file_extension = os.path.splitext(value["path"].split("::")[0])[1] or None
        else:
            audio_file_extension = None
    elif isinstance(value, dict) and "bytes" in value and isinstance(value["bytes"], bytes):
        audio_file_extension = None
    elif isinstance(value, AudioDecoder):
        audio_file_extension = None
    else:
        raise ValueError(
            f"An audio sample should have 'path' and 'bytes' (or be an AudioDecoder) but got {', '.join(value)}."
        )
    return audio_file_extension


def _infer_file_extension(
    file_bytes: bytes, magic_number_collection: dict[str, list[Union[bytes, list[tuple[bytes, int]]]]]
) -> Optional[str]:
    for audio_file_extension, magic_numbers in magic_number_collection.items():
        for magic_number in magic_numbers:
            if not isinstance(magic_number, list):
                magic_number = [(magic_number, 0)]
            if all(file_bytes.startswith(*mn if isinstance(mn, tuple) else mn) for mn in magic_number):
                return audio_file_extension
    return None


def infer_audio_file_extension(audio_file_bytes: bytes) -> Optional[str]:
    return _infer_file_extension(audio_file_bytes, AUDIO_FILE_MAGIC_NUMBERS)


def infer_video_file_extension(video_file_bytes: bytes) -> Optional[str]:
    return _infer_file_extension(video_file_bytes, VIDEO_FILE_MAGIC_NUMBERS)


def video(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    storage_client: StorageClient,
    hf_endpoint: str,
    json_path: Optional[list[Union[str, int]]] = None,
) -> Optional[VideoSource]:
    if datasets.config.TORCHCODEC_AVAILABLE:
        from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]

        _VideoDecoder: Optional[type] = VideoDecoder
    else:
        _VideoDecoder = None

    if value is None:
        return None
    if (
        _VideoDecoder
        and isinstance(value, _VideoDecoder)
        and hasattr(value, "_hf_encoded")
        and isinstance(value._hf_encoded, dict)
    ):
        value = value._hf_encoded  # `datasets` patches `torchcodec` to store the encoded data here
    elif isinstance(value, dict):
        value = {"path": value.get("path"), "bytes": value["bytes"]}
    elif isinstance(value, bytes):
        value = {"path": None, "bytes": value}
    elif isinstance(value, str):
        value = {"path": value, "bytes": None}

    if not (isinstance(value, dict) and "path" in value and "bytes" in value):
        raise TypeError(
            "Video cell must be an encoded dict of a video, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )

    if "path" in value and isinstance(value["path"], str) and value.get("bytes") is None:
        if value["path"].startswith(f"hf://datasets/{dataset}@"):
            src = hf_to_https_url(value["path"], hf_endpoint)
            return VideoSource(src=src)
        if is_local_path(value["path"]):
            return None
        else:
            raise ValueError(f"invalid path: {value['path']}")
    elif "bytes" in value and isinstance(value["bytes"], bytes):
        video_file_bytes = value["bytes"]
    else:
        raise ValueError(
            "Video cell dict must be an video sample, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )

    video_file_extension = get_video_file_extension(value)
    if not video_file_extension:
        video_file_extension = infer_audio_file_extension(video_file_bytes) or ".unknown"
    return create_video_file(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=featureName,
        filename=f"{append_hash_suffix('video', json_path)}{video_file_extension}",
        encoded_video=value,
        storage_client=storage_client,
    )


def get_video_file_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and "bytes" in value and isinstance(value["bytes"], bytes):
        video_file_bytes = value["bytes"]
    elif (
        isinstance(value, dict)
        and "path" in value
        and isinstance(value["path"], str)
        and os.path.exists(value["path"])
    ):
        with open(value["path"], "rb") as f:
            video_file_bytes = f.read()
    else:
        raise ValueError(f"A video sample should have 'path' and 'bytes' but got {', '.join(value)}.")
    return video_file_bytes


def get_video_file_extension(value: Any) -> Optional[str]:
    if isinstance(value, dict) and "path" in value and isinstance(value["path"], str):
        # .split("::")[0] for chained URLs like zip://audio.wav::https://foo.bar/data.zip
        # It might be "" for audio files downloaded from the Hub: make it None
        video_file_extension = os.path.splitext(value["path"].split("::")[0])[1] or None
    elif isinstance(value, dict) and "bytes" in value and isinstance(value["bytes"], bytes):
        video_file_extension = None
    else:
        raise ValueError(f"A video sample should have 'path' and 'bytes' but got {', '.join(value)}.")
    return video_file_extension


def pdf(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    storage_client: StorageClient,
    hf_endpoint: str,
    hf_token: Optional[str],
    json_path: Optional[list[Union[str, int]]] = None,
) -> Optional[PDFSource]:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("bytes"):
        value = pdfplumber.open(BytesIO(value["bytes"]))
    elif isinstance(value, bytes):
        value = pdfplumber.open(BytesIO(value))
    elif isinstance(value, dict) and "path" in value and isinstance(value["path"], str):
        if is_local_path(value["path"]):
            return None
        elif value["path"].startswith(f"hf://datasets/{dataset}@"):
            f = HfFileSystem(endpoint=hf_endpoint, token=hf_token).open(value["path"], "rb")
            value = pdfplumber.open(f)

    if not isinstance(value, pdfplumber.pdf.PDF):
        raise TypeError(
            "PDF cell must be a pdfplumber.pdf.PDF object or an encoded dict of a PDF, "
            f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
        )

    # this function can raise, we don't catch it
    return create_pdf_file(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=featureName,
        pdf=value,
        storage_client=storage_client,
        filename=f"{append_hash_suffix('document', json_path)}.pdf",
    )


def _infer_mesh_file_type(path: Optional[str]) -> Optional[str]:
    """Guess the type of a mesh file from its extension, or None if it's not a supported one.

    The supported types are the ones the viewer renders: glTF binaries, ply, stl and obj files, which
    trimesh reads without additional dependency, and the SOG files of Gaussian splats (see _load_sog).
    """
    if path is None:
        return None
    # the path can be chained with "::" (a file inside an archive), or end with a query string (an URL)
    path_without_query = path.split("::", 1)[0].split("?", 1)[0]
    extension = os.path.splitext(path_without_query)[1].lower().lstrip(".")
    return extension if extension in SUPPORTED_MESH_FILE_TYPES else None


def _load_sog(file: Any) -> "trimesh.PointCloud":
    """Load a SOG file (PlayCanvas' compressed Gaussian splats) as the point cloud of a 3DGS ply file.

    A SOG file is a ZIP of a meta.json and WebP images, where each pixel holds a property of a splat
    (https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/). Its splats are
    decoded to the vertex properties of a 3DGS ply file, in the same raw ply metadata that trimesh
    keeps, so that the thumbnail draws them as it draws the splats of a ply file. Only the splats a
    thumbnail draws are decoded: a random subset of them in large scenes.
    """
    import trimesh
    from PIL import Image

    with zipfile.ZipFile(file) as archive:

        def read_member(name: str) -> bytes:
            if archive.getinfo(name).file_size > SOG_MAX_MEMBER_BYTES:
                raise ValueError(f"The file {name} of the SOG archive is too large.")
            return archive.read(name)

        meta = json.loads(read_member("meta.json"))
        version, count = meta.get("version"), meta["count"]
        if version not in (2, 3):
            raise ValueError(f"Unsupported SOG version: {version}. Only versions 2 and 3 are supported.")
        subset = (
            np.sort(np.random.default_rng(0).choice(count, MESH_THUMBNAIL_MAX_POINTS, replace=False))
            if count > MESH_THUMBNAIL_MAX_POINTS
            else slice(None)
        )

        def read_pixels(name: str) -> np.ndarray:
            with Image.open(BytesIO(read_member(name))) as image:
                if image.width * image.height > SOG_MAX_PIXELS:
                    raise ValueError(f"The image {name} of the SOG archive is too large.")
                # the writers size the images to the scene: decoding an image much larger than it costs memory
                # for nothing (about 1 GiB for 8192x8192 pixels), and we check its size before decoding it
                if not count <= image.width * image.height <= SOG_MAX_PIXELS_PER_SPLAT * count + SOG_MAX_PADDING:
                    raise ValueError(
                        f"The image {name} of the SOG archive has {image.width}x{image.height} pixels, which does not"
                        f" match the {count} splats of the scene."
                    )
                # the splats are the first pixels: only convert the rows that hold them, so that the arrays we
                # keep are the size of the scene, not of the image
                rows = -(-count // image.width)
                pixels = np.asarray(image.crop((0, 0, image.width, rows)).convert("RGBA"))
                return pixels.reshape(-1, 4)[:count][subset]

        means_lower, means_upper = (read_pixels(name) for name in meta["means"]["files"])
        scales, quats, sh0 = (read_pixels(meta[name]["files"][0]) for name in ("scales", "quats", "sh0"))

    # 16-bit positions, quantized in a symmetric log domain
    mins, maxs = np.array(meta["means"]["mins"]), np.array(meta["means"]["maxs"])
    logged = mins + (maxs - mins) * (means_upper[:, :3].astype(np.int32) << 8 | means_lower[:, :3]) / 65535
    positions = np.sign(logged) * np.expm1(np.abs(logged))
    if version == 3:
        # the exponential map of the quaternion, scaled so that a half turn is on the unit sphere
        axis = (quats[:, :3].astype(np.float64) - 128) / 127
        length = np.linalg.norm(axis, axis=1)
        half_angle = np.pi / 2 * length
        rotations = np.column_stack(
            [np.cos(half_angle), axis * (np.sin(half_angle) / np.maximum(length, 1e-12))[:, None]]
        )
        opacities = (sh0[:, 3] / 204) ** 2
    else:
        # "smallest three": the alpha byte tells which (w, x, y, z) component is left out, the largest one
        if (quats[:, 3] < 252).any():
            raise ValueError("Invalid SOG quaternion: the alpha byte of the quats image must be between 252 and 255.")
        kept = (quats[:, :3] / 255 - 0.5) * np.sqrt(2)
        largest = np.sqrt(np.maximum(0.0, 1 - (kept**2).sum(axis=1)))
        rotations = np.insert(kept, 0, largest, axis=1)
        for omitted in range(1, 4):
            rows = quats[:, 3] == 252 + omitted
            rotations[rows] = np.insert(kept[rows], omitted, largest[rows], axis=1)
        opacities = sh0[:, 3] / 255
    # the ply files store the logit of the opacity, and opacities of SOG version 3 can go above 1
    opacities = np.clip(opacities, 1e-6, 1 - 1e-6)
    vertex = {
        "opacity": np.log(opacities / (1 - opacities)),
        **{f"f_dc_{i}": np.array(meta["sh0"]["codebook"])[sh0[:, i]] for i in range(3)},
        **{f"scale_{i}": np.array(meta["scales"]["codebook"])[scales[:, i]] for i in range(3)},
        **{f"rot_{i}": rotations[:, i] for i in range(4)},
    }
    point_cloud = trimesh.PointCloud(positions)  # type: ignore[no-untyped-call]
    point_cloud.metadata["_ply_raw"] = {"vertex": {"data": vertex}}
    return point_cloud


def _load_mesh_scene(file: Any, file_type: str) -> "trimesh.Scene":
    """Load the 3D scene of a mesh file of one of the supported types."""
    import trimesh

    return _as_mesh_scene(_load_sog(file) if file_type == "sog" else trimesh.load(file, file_type=file_type))


def _as_mesh_scene(value: Any) -> "trimesh.Scene":
    """Return the trimesh scene of a decoded mesh cell, which can be a single mesh or a whole scene."""
    import trimesh

    # a ply file without faces, like a point cloud or a Gaussian splat, is loaded as a point cloud
    if isinstance(value, (trimesh.Trimesh, trimesh.PointCloud)):
        return value.scene()
    if isinstance(value, trimesh.Scene):
        return value
    raise TypeError(
        "Mesh cell must be a `trimesh.Scene` object or an encoded dict of a mesh, "
        f"but got {str(value)[:300]}{'...' if len(str(value)) > 300 else ''}"
    )


def mesh(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    value: Any,
    featureName: str,
    storage_client: StorageClient,
    hf_endpoint: str,
    hf_token: Optional[str],
    json_path: Optional[list[Union[str, int]]] = None,
) -> Optional[MeshSource]:
    """Prepare a mesh cell: a rendered thumbnail, and a URL to fetch the 3D file.

    The cell can be:
    - an encoded dict with the mesh bytes, and a path to infer the file type: the mesh file is stored
      on the assets storage with its thumbnail, and "src" is the URL of the stored mesh file;
    - an encoded dict with a path of the dataset on the Hub: the mesh file is fetched from the Hub by
      the viewer, so only its thumbnail is stored, and "src" is its URL on the Hub;
    - a `trimesh.Trimesh` or `trimesh.Scene` object: it's exported to glTF binary, then stored as the
      mesh bytes above.

    It raises a ValueError if the bytes cannot be decoded, or if the path points elsewhere than in the
    dataset. Like the other media cells, a local path is skipped: the file is on the machine of the
    creator of the dataset, and cannot be fetched by the viewer. A file larger than MESH_MAX_FILE_SIZE
    is skipped too: its thumbnail takes too much memory to render.
    """
    if value is None:
        return None
    # the names of the stored assets: "mesh.glb" and "mesh.glb.png", with a suffix if the cell is nested
    filename = append_hash_suffix("mesh", json_path)

    if isinstance(value, bytes):
        # the mesh bytes alone don't tell the file type: it's read from the extension of "path"
        raise ValueError(MESH_FILE_TYPE_ERROR)

    if isinstance(value, dict) and value.get("bytes") is not None:
        mesh_file_type = _infer_mesh_file_type(value.get("path"))
        if mesh_file_type is None:
            raise ValueError(MESH_FILE_TYPE_ERROR)
        mesh_bytes: bytes = value["bytes"]
        if len(mesh_bytes) > MESH_MAX_FILE_SIZE:
            return None
        # this function can raise, we don't catch it
        return create_mesh_file(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            column=featureName,
            filename=f"{filename}.{mesh_file_type}",
            mesh_bytes=mesh_bytes,
            load_mesh=lambda: _load_mesh_scene(BytesIO(mesh_bytes), mesh_file_type),
            storage_client=storage_client,
        )

    if isinstance(value, dict) and isinstance(value.get("path"), str):
        path = value["path"]
        if is_local_path(path):
            return None
        if not path.startswith(f"hf://datasets/{dataset}@"):
            # only the files of the dataset itself can be fetched, and served by the viewer
            raise ValueError(f"invalid path: {path}")
        file_type = _infer_mesh_file_type(path)
        if file_type is None:
            raise ValueError(MESH_FILE_TYPE_ERROR)
        fs = HfFileSystem(endpoint=hf_endpoint, token=hf_token)
        size_bytes = fs.size(path)
        if size_bytes > MESH_MAX_FILE_SIZE:
            return None

        def load_mesh() -> "trimesh.Scene":
            with fs.open(path, "rb") as file:
                return _load_mesh_scene(file, file_type)

        # the mesh file stays on the Hub: it's already available at its own URL, we don't store a copy
        # this function can raise, we don't catch it
        thumbnail = create_mesh_thumbnail(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            column=featureName,
            filename=f"{filename}.{file_type}",
            load_mesh=load_mesh,
            storage_client=storage_client,
        )
        return MeshSource(
            src=hf_to_https_url(path, hf_endpoint),
            size_bytes=size_bytes,
            thumbnail=thumbnail,
        )

    # a mesh or a scene object: it has not been encoded for storage, so we export it to glTF binary
    scene = _as_mesh_scene(value)
    return create_mesh_file(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=featureName,
        filename=f"{filename}.glb",
        mesh_bytes=scene.export(file_type="glb"),  # type: ignore
        load_mesh=lambda: scene,
        storage_client=storage_client,
    )


def get_cell_value(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    cell: Any,
    featureName: str,
    fieldType: Any,
    storage_client: StorageClient,
    hf_endpoint: str,
    hf_token: Optional[str],
    json_path: Optional[list[Union[str, int]]] = None,
) -> Any:
    # always allow None values in the cells
    if cell is None:
        return cell
    if isinstance(fieldType, Image):
        return image(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            storage_client=storage_client,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            json_path=json_path,
        )
    elif isinstance(fieldType, Audio):
        return audio(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            storage_client=storage_client,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            json_path=json_path,
        )
    elif isinstance(fieldType, Video):
        return video(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            storage_client=storage_client,
            hf_endpoint=hf_endpoint,
            json_path=json_path,
        )
    elif isinstance(fieldType, Pdf):
        return pdf(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            storage_client=storage_client,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            json_path=json_path,
        )
    elif isinstance(fieldType, Mesh):
        return mesh(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=row_idx,
            value=cell,
            featureName=featureName,
            storage_client=storage_client,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            json_path=json_path,
        )
    elif isinstance(fieldType, Json):
        try:
            return fieldType.decode_example(cell)
        except (ValueError, TypeError):
            return cell
    elif isinstance(fieldType, list):
        if not isinstance(cell, list):
            raise TypeError("list cell must be a list.")
        if len(fieldType) != 1:
            raise TypeError("the feature type should be a 1-element list.")
        subFieldType = fieldType[0]
        return [
            get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=subCell,
                featureName=featureName,
                fieldType=subFieldType,
                storage_client=storage_client,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                json_path=json_path + [idx] if json_path else [idx],
            )
            for (idx, subCell) in enumerate(cell)
        ]
    elif isinstance(fieldType, LargeList):
        if not isinstance(cell, list):
            raise TypeError("list cell must be a list.")
        subFieldType = fieldType.feature
        return [
            get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=subCell,
                featureName=featureName,
                fieldType=subFieldType,
                storage_client=storage_client,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                json_path=json_path + [idx] if json_path else [idx],
            )
            for (idx, subCell) in enumerate(cell)
        ]
    elif isinstance(fieldType, List):
        if not isinstance(cell, list):
            raise TypeError("list cell must be a list.")
        if fieldType.length >= 0 and len(cell) != fieldType.length:
            raise TypeError("the cell length should be the same as the List length.")
        return [
            get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=subCell,
                featureName=featureName,
                fieldType=fieldType.feature,
                storage_client=storage_client,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                json_path=json_path + [idx] if json_path else [idx],
            )
            for (idx, subCell) in enumerate(cell)
        ]

    elif isinstance(fieldType, dict):
        if not isinstance(cell, dict):
            raise TypeError("dict cell must be a dict.")
        return {
            key: get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=subCell,
                featureName=featureName,
                fieldType=fieldType[key],
                storage_client=storage_client,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                json_path=json_path + [key] if json_path else [key],
            )
            for (key, subCell) in cell.items()
        }
    elif isinstance(
        fieldType,
        (
            Value,
            ClassLabel,
            Array2D,
            Array3D,
            Array4D,
            Array5D,
            Translation,
            TranslationVariableLanguages,
        ),
    ):
        return cell
    else:
        raise TypeError("could not determine the type of the data cell.")


# in JSON, dicts do not carry any order, so we need to return a list
#
# > An object is an *unordered* collection of zero or more name/value pairs, where a name is a string and a value
#   is a string, number, boolean, null, object, or array.
# > An array is an *ordered* sequence of zero or more values.
# > The terms "object" and "array" come from the conventions of JavaScript.
# from https://stackoverflow.com/a/7214312/7351594 / https://www.rfc-editor.org/rfc/rfc7159.html
def to_features_list(features: Features) -> list[FeatureItem]:
    features_dict = features.to_dict()
    return [
        {
            "feature_idx": idx,
            "name": name,
            "type": features_dict[name],
        }
        for idx, name in enumerate(features)
    ]
