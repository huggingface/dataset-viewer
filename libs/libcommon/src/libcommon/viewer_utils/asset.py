# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import os
import threading
from collections.abc import Hashable
from io import BufferedReader, BytesIO
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union
from unittest.mock import patch

import fitz
import numpy as np
from datasets import Audio

if TYPE_CHECKING:
    import trimesh
    from pdfplumber.pdf import PDF
    from PIL import Image

    from libcommon.storage_client import StorageClient


SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
}
SUPPORTED_AUDIO_EXTENSIONS = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE.keys()
DATASET_GIT_REVISION_PLACEHOLDER = "{dataset_git_revision}"
LOCK = threading.Lock()
# the size of the rendered 3D thumbnails, in pixels
MESH_THUMBNAIL_WIDTH = 800
MESH_THUMBNAIL_HEIGHT = 600

logger = logging.getLogger(__name__)


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
    size_bytes: int
    thumbnail: ImageSource


class MeshSource(TypedDict):
    src: str
    size_bytes: int
    thumbnail: ImageSource


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
            from torchcodec.encoders import AudioEncoder  # type: ignore[attr-defined]

            with NamedTemporaryFile("wb", suffix=audio_file_extension, delete_on_close=False) as tmpfile:
                tmpfile.write(audio_file_bytes)
                tmpfile.close()
                audio = Audio().decode_example({"path": tmpfile.name, "bytes": None})
                samples = audio.get_all_samples()
                buffer = BytesIO()
                num_channels = samples.data.shape[0]
                AudioEncoder(samples.data.cpu(), sample_rate=samples.sample_rate).to_file_like(
                    buffer, format="wav", num_channels=num_channels
                )
                with storage_client._fs.open(audio_path, "wb") as f:
                    f.write(buffer.getvalue())
    return [AudioSource(src=storage_client.get_url(object_path, revision=revision), type=media_type)]


def create_pdf_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    pdf: "PDF",
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
    thumbnail_buffer = BytesIO()
    pdf.stream.seek(0)
    with fitz.open(stream=pdf.stream.read(), filetype="pdf") as doc:
        thumbnail = doc.load_page(0).get_pixmap().pil_image()
        thumbnail.save(thumbnail_buffer, format="PNG")
    thumbnail_buffer.seek(0)

    if storage_client.overwrite or not storage_client.exists(thumbnail_storage_path):
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
            return b"%PDF" in pdf_stream.read(1000)
        finally:
            pdf_stream.seek(current_position)

    pdf_data = pdf.stream
    if not is_valid_pdf(pdf_data):
        raise ValueError("The provided data is not a valid PDF.")

    if storage_client.overwrite or not storage_client.exists(pdf_storage_path):
        with storage_client._fs.open(storage_client.get_full_path(pdf_storage_path), "wb") as pdf_file:
            pdf_data.seek(0)
            pdf_file.write(pdf_data.read())

    # Get the size of the PDF file, probably not needed
    pdf_data.seek(0, 2)
    size_bytes = pdf_data.tell()
    pdf_data.seek(0)

    return PDFSource(
        src=storage_client.get_url(pdf_object_path, revision=revision),
        size_bytes=size_bytes,
        thumbnail=ImageSource(
            src=storage_client.get_url(thumbnail_object_path, revision=revision),
            height=thumbnail.height,
            width=thumbnail.width,
        ),
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
    if "bytes" in encoded_video and isinstance(encoded_video["bytes"], bytes):
        # (rare and not very important) we attempt to upload video data from webdataset/parquet files but don't process them
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
    elif (
        encoded_video.get("bytes") is None
        and "path" in encoded_video
        and isinstance(encoded_video["path"], str)
        and encoded_video["path"].startswith(f"hf://datasets/{dataset}@")
    ):
        # in general video files are stored in the dataset repository, we can just get the URL
        # (`datasets` doesn't embed the video bytes in Parquet when the file is already on HF)
        object_path = encoded_video["path"].replace(revision, DATASET_GIT_REVISION_PLACEHOLDER)
    else:
        raise ValueError("The video cell doesn't contain a valid path or bytes")
    src = storage_client.get_url(object_path, revision=revision)
    return VideoSource(src=src)


def _robust_format_texture_source(
    texture: Optional[Union["Image.Image", np.ndarray]], target_channels: str = "RGB"
) -> Optional[np.ndarray]:
    """Format a texture as a np array of uint8, where pyrender's version of this function fails.

    It replaces pyrender's format_texture_source (a copy of it, with the same signature and the same
    returned value) for the textures that trimesh extracts from a 3D file, in two cases:
    1. the boolean textures: pyrender does not know the boolean dtype, and raises a TypeError for them
       (it only tests floating and integer dtypes). A boolean mask is rendered black and white.
    2. the 2-channels textures: pyrender refuses to convert them to RGBA. We fill the missing channel
       with the average of the two others, and the alpha channel with 255 (fully opaque).
    """
    import numpy as np
    from PIL import Image as PILImage

    # Pass through None
    if texture is None:
        return None

    # Convert PIL images into numpy arrays
    if isinstance(texture, PILImage.Image):
        if texture.mode == "P" and target_channels in ("RGB", "RGBA"):
            texture = np.array(texture.convert(target_channels))
        else:
            texture = np.array(texture)

    # Format numpy arrays
    if isinstance(texture, np.ndarray):
        if np.issubdtype(texture.dtype, np.floating):
            texture = np.array(texture * 255.0, dtype=np.uint8)
        elif np.issubdtype(texture.dtype, np.bool_):
            # a boolean mask is a black and white image: 0 and 255, not 0 and 1
            texture = texture.astype(np.uint8) * 255
        elif np.issubdtype(texture.dtype, np.integer):
            texture = texture.astype(np.uint8)
        else:
            raise TypeError("Invalid type {} for texture".format(texture.dtype))

        # Format array by picking out correct texture channels or padding
        if texture.ndim == 2:
            texture = texture[:, :, np.newaxis]
        if target_channels == "R":
            texture = texture[:, :, 0]
            texture = texture.squeeze()
        elif target_channels == "RG":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 2, axis=2)
            else:
                texture = texture[:, :, (0, 1)]
        elif target_channels == "GB":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 2, axis=2)
            elif texture.shape[2] > 2:
                texture = texture[:, :, (1, 2)]
        elif target_channels == "RGB":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 3, axis=2)
            elif texture.shape[2] == 2:
                raise ValueError("Cannot reformat 2-channel texture into RGB")
            else:
                texture = texture[:, :, (0, 1, 2)]
        elif target_channels == "RGBA":
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 4, axis=2)
                texture[:, :, 3] = 255
            elif texture.shape[2] == 2:
                tx = np.empty((texture.shape[0], texture.shape[1], 4), dtype=np.uint8)
                tx[:, :, :2] = texture
                tx[:, :, 2] = texture.mean(axis=2).astype(np.uint8)
                tx[:, :, 3] = 255
                texture = tx
            elif texture.shape[2] == 3:
                tx = np.empty((texture.shape[0], texture.shape[1], 4), dtype=np.uint8)
                tx[:, :, :3] = texture
                tx[:, :, 3] = 255
                texture = tx
        else:
            raise ValueError("Invalid texture channel specification: {}".format(target_channels))
    else:
        raise TypeError("Invalid type {} for texture".format(type(texture)))

    return texture


# The camera of the SimReady thumbnails (NVIDIA's simready-thumbnail): an orthographic three-quarter view
MESH_CAMERA_ELEVATION_DEG = 45.0
MESH_CAMERA_AZIMUTH_DEG = 45.0
# Their key, fill and rim lights, as directions in the camera space (x right, y up, z toward the camera)
MESH_THUMBNAIL_LIGHTS = [((-1.0, 1.0, 1.0), 3.0), ((1.0, 0.0, 1.0), 1.5), ((0.0, 1.0, -1.0), 2.0)]


def _pose_looking_along(direction: np.ndarray, position: np.ndarray) -> np.ndarray:
    """A 4x4 OpenGL pose at the given position, looking along -direction (cameras and lights look along -Z)."""
    z_axis = direction / np.linalg.norm(direction)
    up = np.array([0.0, 1.0, 0.0]) if abs(z_axis[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = np.cross(z_axis, x_axis)
    pose[:3, 2] = z_axis
    pose[:3, 3] = position
    return pose


def _compute_scene_camera_pose(scene: "trimesh.Scene") -> tuple[np.ndarray, float, float, float]:
    """
    Computes the 4x4 OpenGL pose of an orthographic camera looking at the center of the scene bounds from the
    SimReady three-quarter view, the half size of a view that contains the whole scene, and the clipping planes.
    """
    if scene.is_empty:
        return np.eye(4), 1.0, 0.05, 100.0

    center = scene.bounds.mean(axis=0)
    radius = float(np.linalg.norm(scene.extents) / 2.0)
    elevation, azimuth = np.radians(MESH_CAMERA_ELEVATION_DEG), np.radians(MESH_CAMERA_AZIMUTH_DEG)
    view_direction = np.array(
        [np.cos(elevation) * np.cos(azimuth), np.sin(elevation), np.cos(elevation) * np.sin(azimuth)]
    )
    distance = radius * 4.0
    camera_pose = _pose_looking_along(view_direction, center + view_direction * distance)
    return camera_pose, radius, distance - radius * 1.1, distance + radius * 1.1


# The thumbnail is rendered larger, then cropped to the object: it fills the frame whatever its shape
MESH_THUMBNAIL_SUPERSAMPLING = 2
MESH_THUMBNAIL_MARGIN = 0.05
# Stands in for the dome of the SimReady setup: without it, metallic surfaces have nothing to reflect, and are black
MESH_THUMBNAIL_AMBIENT_LIGHT = 0.3


def _fit_to_frame(image: "Image.Image", width: int, height: int) -> "Image.Image":
    """Crop the image to its non-transparent pixels, and center them in a transparent frame of the given size."""
    from PIL import Image as PILImage

    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return image.resize((width, height), PILImage.Resampling.LANCZOS)
    cropped = image.crop(bbox)
    scale = min(
        width * (1 - 2 * MESH_THUMBNAIL_MARGIN) / cropped.width,
        height * (1 - 2 * MESH_THUMBNAIL_MARGIN) / cropped.height,
    )
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))), PILImage.Resampling.LANCZOS
    )
    frame = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
    frame.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return frame


def _render_trimesh_scene_to_png(
    trimesh_scene: "trimesh.Scene",
    output_buffer: BytesIO,
    width: int = MESH_THUMBNAIL_WIDTH,
    height: int = MESH_THUMBNAIL_HEIGHT,
) -> None:
    """
    Converts a trimesh Scene to pyrender, applies the camera heuristic,
    and saves the output to a transparent PNG image.

    It needs a headless OpenGL context: pyrender uses EGL or OSMesa, depending on the
    PYOPENGL_PLATFORM environment variable ("egl" or "osmesa"). Only EGL gives an alpha channel, so
    the background is transparent with it, and opaque black with OSMesa.
    """
    # pyrender needs an offscreen OpenGL context, and falls back to pyglet (which needs a display, so
    # it fails on a server) unless the platform is set: default to EGL, the only one that gives an
    # alpha channel, unless the deployment already chose another one (e.g "osmesa"). It must be set
    # before pyrender is imported, as PyOpenGL reads it when it's imported.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import numpy as np
    import pyrender  # type: ignore
    from PIL import Image as PILImage

    # 1. Convert trimesh.Scene to pyrender.Scene
    #
    # pyrender formats the textures with its own format_texture_source, which refuses some of the
    # textures trimesh extracts from a 3D file (see _robust_format_texture_source). As the name is
    # looked up in the module where it's used, both call sites are patched.
    geometries: dict[Hashable, Any] = {}
    with (
        patch("pyrender.material.format_texture_source", _robust_format_texture_source),
        patch("pyrender.texture.format_texture_source", _robust_format_texture_source),
    ):
        for name, geom in trimesh_scene.geometry.items():
            try:
                geometries[name] = pyrender.Mesh.from_trimesh(geom)
            except Exception:
                # a geometry we cannot render (a point cloud, for instance) must not prevent us from
                # rendering the rest of the scene
                logger.warning(f"Skipping the geometry {name} while rendering a 3D thumbnail.", exc_info=True)
    if not geometries:
        raise ValueError("The 3D scene doesn't contain any renderable geometry.")

    # create the pyrender scene object
    pyrender_scene = pyrender.Scene(ambient_light=[MESH_THUMBNAIL_AMBIENT_LIGHT] * 3)

    # add every node with a renderable geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        if geom_name in geometries:
            pyrender_scene.add(geometries[geom_name], pose=pose)

    # 2. Set background color to fully transparent [R, G, B, Alpha]
    pyrender_scene.bg_color = np.array([0.0, 0.0, 0.0, 0.0])  # 0.0 alpha means transparent

    # 3. Configure the orthographic camera
    aspect_ratio = float(width) / float(height)

    # Compute heuristic camera pose
    camera_pose, half_size, znear, zfar = _compute_scene_camera_pose(trimesh_scene)

    camera = pyrender.OrthographicCamera(xmag=half_size * aspect_ratio, ymag=half_size, znear=znear, zfar=zfar)
    pyrender_scene.add(camera, pose=camera_pose)

    # 4. Add the key, fill and rim lights
    for direction, intensity in MESH_THUMBNAIL_LIGHTS:
        world_direction = camera_pose[:3, :3] @ np.array(direction)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        pyrender_scene.add(light, pose=_pose_looking_along(world_direction, camera_pose[:3, 3]))

    # 5. Initialize the Offscreen Renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=width * MESH_THUMBNAIL_SUPERSAMPLING, viewport_height=height * MESH_THUMBNAIL_SUPERSAMPLING
    )

    try:
        # Render the scene. The RGBA flag asks for the alpha channel: without it, pyrender always
        # returns an RGB image, and the background cannot be transparent.
        color_image, _ = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.RGBA)

        # Save the image using Pillow as RGBA. Some platforms (e.g OSMesa) return an RGB image even
        # with the RGBA flag: in that case the background is opaque, and only the colors are kept.
        if color_image.shape[2] == 3:
            img = PILImage.fromarray(color_image).convert("RGBA").resize((width, height), PILImage.Resampling.LANCZOS)
        else:
            img = _fit_to_frame(PILImage.fromarray(color_image, mode="RGBA"), width, height)

        img.save(output_buffer, "PNG")

    finally:
        renderer.delete()


def create_mesh_thumbnail(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    mesh: "trimesh.Scene",
    storage_client: "StorageClient",
) -> ImageSource:
    """Render a 3D scene to a PNG image, and store it on the assets storage.

    The image is not resized to a thumbnail size: it's rendered at the requested size, so that the
    edges stay sharp.
    """
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
    thumbnail_buffer = BytesIO()
    thumbnail_width = MESH_THUMBNAIL_WIDTH
    thumbnail_height = MESH_THUMBNAIL_HEIGHT
    _render_trimesh_scene_to_png(mesh, thumbnail_buffer, width=thumbnail_width, height=thumbnail_height)
    thumbnail_buffer.seek(0)

    if storage_client.overwrite or not storage_client.exists(thumbnail_storage_path):
        with storage_client._fs.open(storage_client.get_full_path(thumbnail_storage_path), "wb") as thumbnail_file:
            thumbnail_file.write(thumbnail_buffer.read())

    return ImageSource(
        src=storage_client.get_url(thumbnail_object_path, revision=revision),
        height=thumbnail_height,
        width=thumbnail_width,
    )


def create_mesh_file(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    row_idx: int,
    column: str,
    filename: str,
    mesh: "trimesh.Scene",
    mesh_bytes: bytes,
    storage_client: "StorageClient",
) -> MeshSource:
    """Store a mesh cell on the assets storage: the mesh file, and a rendered thumbnail of it.

    The mesh file is stored as well, and not only the thumbnail: the viewers fetch the 3D file to
    display it, to download it, or to rotate it. A cell whose mesh is already available at its own
    URL does not go through this function: only its thumbnail is created (see create_mesh_thumbnail).
    """
    thumbnail = create_mesh_thumbnail(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
        mesh=mesh,
        storage_client=storage_client,
    )
    mesh_object_path = storage_client.generate_object_path(
        dataset=dataset,
        revision=DATASET_GIT_REVISION_PLACEHOLDER,
        config=config,
        split=split,
        row_idx=row_idx,
        column=column,
        filename=filename,
    )
    mesh_storage_path = replace_dataset_git_revision_placeholder(mesh_object_path, revision=revision)
    if storage_client.overwrite or not storage_client.exists(mesh_storage_path):
        with storage_client._fs.open(storage_client.get_full_path(mesh_storage_path), "wb") as mesh_file:
            mesh_file.write(mesh_bytes)

    return MeshSource(
        src=storage_client.get_url(mesh_object_path, revision=revision),
        size_bytes=len(mesh_bytes),
        thumbnail=thumbnail,
    )
