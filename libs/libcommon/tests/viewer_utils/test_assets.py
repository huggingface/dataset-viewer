import os.path
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pymupdf
import pytest
import trimesh
import validators  # type: ignore
from pdfplumber import open
from PIL import Image as PILImage

from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.asset import (
    DATASET_GIT_REVISION_PLACEHOLDER,
    MESH_THUMBNAIL_HEIGHT,
    MESH_THUMBNAIL_WIDTH,
    SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE,
    ImageSource,
    _render_trimesh_scene_to_png,
    _robust_format_texture_source,
    create_audio_file,
    create_image_file,
    create_mesh_file,
    create_mesh_thumbnail,
    create_pdf_file,
    create_video_file,
)

from ..constants import (
    ASSETS_BASE_URL,
    CI_HUB_ENDPOINT,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_ROW_IDX,
    DEFAULT_SPLIT,
)
from ..mesh_rendering import ensure_mesh_rendering
from ..types import DatasetFixture


def test_create_image_file(
    storage_client_with_url_preparator: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture]
) -> None:
    dataset_name = "image"
    dataset_fixture = datasets_fixtures[dataset_name]
    value = create_image_file(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        image=dataset_fixture.dataset[DEFAULT_ROW_IDX][DEFAULT_COLUMN_NAME],
        column=DEFAULT_COLUMN_NAME,
        filename="image.jpg",
        row_idx=DEFAULT_ROW_IDX,
        format="JPEG",
        storage_client=storage_client_with_url_preparator,
    )
    assert value == dataset_fixture.expected_cell
    image_key = value["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    image_path = image_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client_with_url_preparator.exists(image_path)

    image = PILImage.open(storage_client_with_url_preparator.get_full_path(image_path))
    assert image is not None


@pytest.mark.parametrize("audio_file_extension", ["WITH_EXTENSION", "WITHOUT_EXTENSION"])
@pytest.mark.parametrize("audio_file_name", ["test_audio_44100.wav", "test_audio_opus.opus"])
@pytest.mark.parametrize("filename_extension", [".wav", ".opus"])
def test_create_audio_file(
    audio_file_name: str,
    audio_file_extension: str,
    filename_extension: str,
    shared_datadir: Path,
    storage_client_with_url_preparator: StorageClient,
) -> None:
    audio_file_extension = os.path.splitext(audio_file_name)[1] if audio_file_extension == "WITH_EXTENSION" else ""
    audio_file_bytes = (shared_datadir / audio_file_name).read_bytes()
    filename = "audio" + filename_extension
    mime_type = SUPPORTED_AUDIO_EXTENSION_TO_MEDIA_TYPE[filename_extension]
    value = create_audio_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        audio_file_extension=audio_file_extension,
        audio_file_bytes=audio_file_bytes,
        column="col",
        filename=filename,
        storage_client=storage_client_with_url_preparator,
    )
    audio_key = "dataset/--/revision/--/config/split/7/col/" + filename
    assert value == [
        {
            "src": f"{ASSETS_BASE_URL}/{audio_key}",
            "type": mime_type,
        },
    ]
    assert storage_client_with_url_preparator.exists(audio_key)


@pytest.mark.parametrize(
    "pdf_file,expected_size",
    [("test_A4.pdf", 8810), ("test_us_letter.pdf", 1319)],
)
def test_create_pdf_file(
    pdf_file: str,
    expected_size: int,
    shared_datadir: Path,
    storage_client_with_url_preparator: StorageClient,
) -> None:
    pdf = open(shared_datadir / pdf_file)
    value = create_pdf_file(
        dataset="dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        column="col",
        filename=pdf_file,
        pdf=pdf,
        storage_client=storage_client_with_url_preparator,
    )
    pdf_key = value["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    pdf_path = pdf_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client_with_url_preparator.exists(pdf_path)
    new_pdf = open(storage_client_with_url_preparator.get_full_path(pdf_path))
    assert new_pdf is not None

    thumbnail_key = value["thumbnail"]["src"].removeprefix(f"{ASSETS_BASE_URL}/")
    thumbnail_path = thumbnail_key.replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client_with_url_preparator.exists(thumbnail_path)
    image = PILImage.open(storage_client_with_url_preparator.get_full_path(thumbnail_path))
    assert image is not None
    assert image.size == (value["thumbnail"]["width"], value["thumbnail"]["height"])
    assert value["size_bytes"] == expected_size


def test_fixed_mupdf_version() -> None:
    assert pymupdf.mupdf_version_tuple >= (1, 28, 0)


@pytest.mark.parametrize(
    "path",
    [
        None,
        "https://datasets-server.huggingface.co/assets/victim/private/--/revision/--/default/train/0/video.mp4",
        "https://datasets-server.huggingface.co/cached-assets/victim/private/--/revision/--/default/train/0/video.mp4",
    ],
)
def test_create_video_file_stores_embedded_bytes(
    path: str | None, storage_client_with_url_preparator: StorageClient
) -> None:
    video_bytes = b"embedded video bytes"
    value = create_video_file(
        dataset="attacker/dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        column="col",
        filename="video.mp4",
        encoded_video={"path": path, "bytes": video_bytes},
        storage_client=storage_client_with_url_preparator,
    )
    video_key = "attacker/dataset/--/revision/--/config/split/7/col/video.mp4"

    assert value == {"src": f"{ASSETS_BASE_URL}/{video_key}"}
    assert storage_client_with_url_preparator.exists(video_key)
    with storage_client_with_url_preparator._fs.open(
        storage_client_with_url_preparator.get_full_path(video_key), "rb"
    ) as video_file:
        assert video_file.read() == video_bytes


def test_create_video_file_reuses_same_dataset_hf_path(
    storage_client_with_url_preparator: StorageClient,
) -> None:
    value = create_video_file(
        dataset="owner/dataset",
        revision="revision",
        config="config",
        split="split",
        row_idx=7,
        column="col",
        filename="video.mp4",
        encoded_video={"path": "hf://datasets/owner/dataset@revision/video.mp4", "bytes": None},
        storage_client=storage_client_with_url_preparator,
    )

    assert value == {"src": f"{CI_HUB_ENDPOINT}/datasets/owner/dataset/resolve/revision/video.mp4"}


def test_create_video_file_rejects_cross_dataset_hf_path(
    storage_client_with_url_preparator: StorageClient,
) -> None:
    with pytest.raises(ValueError, match="The video cell doesn't contain a valid path or bytes"):
        create_video_file(
            dataset="owner/dataset",
            revision="revision",
            config="config",
            split="split",
            row_idx=7,
            column="col",
            filename="video.mp4",
            encoded_video={"path": "hf://datasets/victim/private@revision/video.mp4", "bytes": None},
            storage_client=storage_client_with_url_preparator,
        )


@pytest.mark.parametrize(
    "dataset,config,split,column",
    [
        ("dataset", "config", "split", "column"),
        ("dataset", "config?<script>alert('XSS');</script>&", "split", "column?"),
    ],
)
def test_src_is_sanitized(storage_client: StorageClient, dataset: str, config: str, split: str, column: str) -> None:
    base_url = "https://datasets-server.huggingface.co/assets"
    filename = "image.jpg"
    object_key = storage_client.generate_object_path(
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        row_idx=0,
        column=column,
        filename=filename,
    )
    src = f"{base_url}/{object_key}"
    assert validators.url(src)


# --- 3D meshes ---

# a scene with four geometries and textured materials, also used by the "mesh" dataset fixture
PATH_TO_MESH_GLB = Path(__file__).resolve().parent.parent / "fixtures" / "data" / "test_mesh.glb"
# where the assets created by the tests below are stored: images, and 3D files
MESH_DATASET = "dataset"
MESH_CONFIG = "config"
MESH_SPLIT = "split"
MESH_ROW_IDX = 7
MESH_ASSET_PATH = (
    f"{MESH_DATASET}/--/{DEFAULT_REVISION}/--/{MESH_CONFIG}/{MESH_SPLIT}/{MESH_ROW_IDX}/{DEFAULT_COLUMN_NAME}"
)


def get_stored_asset_path(storage_client: StorageClient, src: str) -> str:
    """Get the path of a stored asset, from the URL returned to the client."""
    key = src.removeprefix(f"{ASSETS_BASE_URL}/").replace(DATASET_GIT_REVISION_PLACEHOLDER, DEFAULT_REVISION)
    assert storage_client.exists(key), f"{key} is not in the assets storage"
    return storage_client.get_full_path(key)


def assert_thumbnail(thumbnail: ImageSource, storage_client: StorageClient, filename: str) -> np.ndarray:
    """Check that a rendered thumbnail is a PNG image of the expected size, and return its pixels."""
    assert thumbnail == {
        "src": f"{ASSETS_BASE_URL}/{MESH_ASSET_PATH}/{filename}.png",
        "height": MESH_THUMBNAIL_HEIGHT,
        "width": MESH_THUMBNAIL_WIDTH,
    }
    image = PILImage.open(get_stored_asset_path(storage_client, thumbnail["src"]))
    assert image.format == "PNG"
    assert image.size == (MESH_THUMBNAIL_WIDTH, MESH_THUMBNAIL_HEIGHT)
    return np.asarray(image.convert("RGBA"))


@pytest.mark.parametrize("file_type", ["glb", "ply", "stl", "obj"])
def test_create_mesh_thumbnail_file_types(file_type: str, storage_client_with_url_preparator: StorageClient) -> None:
    """Every supported mesh file type is rendered to a thumbnail, framed on the mesh."""
    ensure_mesh_rendering()
    exported = trimesh.creation.box().export(file_type=file_type)
    # the obj files are text: trimesh exports them to a str, while the cells contain bytes
    mesh_bytes = exported.encode() if isinstance(exported, str) else exported
    mesh = trimesh.load(BytesIO(mesh_bytes), file_type=file_type)
    value = create_mesh_thumbnail(
        dataset=MESH_DATASET,
        revision=DEFAULT_REVISION,
        config=MESH_CONFIG,
        split=MESH_SPLIT,
        row_idx=MESH_ROW_IDX,
        column=DEFAULT_COLUMN_NAME,
        filename=f"mesh.{file_type}",
        mesh=trimesh.Scene(mesh),
        storage_client=storage_client_with_url_preparator,
    )
    pixels = assert_thumbnail(value, storage_client_with_url_preparator, filename=f"mesh.{file_type}")
    assert pixels.shape == (MESH_THUMBNAIL_HEIGHT, MESH_THUMBNAIL_WIDTH, 4)
    assert pixels[0, 0, 3] == 0, "the corners are the background: it must be transparent"
    coverage = float((pixels[:, :, 3] > 0).mean())
    assert 0.01 < coverage < 0.95, f"the mesh covers {coverage:.2%} of the thumbnail, in {file_type}"
    rows, columns = np.nonzero(pixels[:, :, 3])
    filled = max(
        (columns.max() - columns.min() + 1) / MESH_THUMBNAIL_WIDTH,
        (rows.max() - rows.min() + 1) / MESH_THUMBNAIL_HEIGHT,
    )
    assert filled > 0.85, f"the mesh fills {filled:.0%} of the thumbnail width or height, in {file_type}"


def test_create_mesh_thumbnail_scene(storage_client_with_url_preparator: StorageClient) -> None:
    """A scene with several geometries, and materials with textures, is rendered to a thumbnail."""
    ensure_mesh_rendering()
    mesh = trimesh.load(PATH_TO_MESH_GLB)
    assert isinstance(mesh, trimesh.Scene)
    assert len(mesh.geometry) > 1
    value = create_mesh_thumbnail(
        dataset=MESH_DATASET,
        revision=DEFAULT_REVISION,
        config=MESH_CONFIG,
        split=MESH_SPLIT,
        row_idx=MESH_ROW_IDX,
        column=DEFAULT_COLUMN_NAME,
        filename="mesh.glb",
        mesh=mesh,
        storage_client=storage_client_with_url_preparator,
    )
    pixels = assert_thumbnail(value, storage_client_with_url_preparator, filename="mesh.glb")
    opaque_pixels = pixels[pixels[:, :, 3] > 0][:, :3]
    assert len(np.unique(opaque_pixels.reshape(-1, 3), axis=0)) > 1, "the geometries are lit and textured"


def test_create_mesh_thumbnail_skips_unsupported_geometry(
    storage_client_with_url_preparator: StorageClient,
) -> None:
    """A geometry pyrender cannot convert (a point cloud) is skipped: the rest of the scene renders."""
    ensure_mesh_rendering()
    box = trimesh.creation.box()
    scene = trimesh.Scene(box)
    scene.add_geometry(trimesh.PointCloud(box.vertices))  # type: ignore
    value = create_mesh_thumbnail(
        dataset=MESH_DATASET,
        revision=DEFAULT_REVISION,
        config=MESH_CONFIG,
        split=MESH_SPLIT,
        row_idx=MESH_ROW_IDX,
        column=DEFAULT_COLUMN_NAME,
        filename="mesh.glb",
        mesh=scene,
        storage_client=storage_client_with_url_preparator,
    )
    pixels = assert_thumbnail(value, storage_client_with_url_preparator, filename="mesh.glb")
    assert float((pixels[:, :, 3] > 0).mean()) > 0.01, "the mesh of the scene is still rendered"


def test_create_mesh_thumbnail_without_geometry(storage_client_with_url_preparator: StorageClient) -> None:
    """A scene without any renderable geometry raises an error, instead of returning a blank image."""
    ensure_mesh_rendering()
    with pytest.raises(ValueError, match="doesn't contain any renderable geometry"):
        create_mesh_thumbnail(
            dataset=MESH_DATASET,
            revision=DEFAULT_REVISION,
            config=MESH_CONFIG,
            split=MESH_SPLIT,
            row_idx=MESH_ROW_IDX,
            column=DEFAULT_COLUMN_NAME,
            filename="mesh.ply",
            mesh=trimesh.Scene([trimesh.PointCloud(np.random.rand(10, 3))]),  # type: ignore
            storage_client=storage_client_with_url_preparator,
        )


def test_create_mesh_file_stores_mesh_and_thumbnail(
    storage_client_with_url_preparator: StorageClient,
) -> None:
    """The mesh file itself is stored as well as its thumbnail, so that it can be fetched and drawn."""
    ensure_mesh_rendering()
    mesh_bytes = PATH_TO_MESH_GLB.read_bytes()
    mesh = trimesh.load(BytesIO(mesh_bytes), file_type="glb")
    assert isinstance(mesh, trimesh.Scene)
    value = create_mesh_file(
        dataset=MESH_DATASET,
        revision=DEFAULT_REVISION,
        config=MESH_CONFIG,
        split=MESH_SPLIT,
        row_idx=MESH_ROW_IDX,
        column=DEFAULT_COLUMN_NAME,
        filename="mesh.glb",
        mesh=mesh,
        mesh_bytes=mesh_bytes,
        storage_client=storage_client_with_url_preparator,
    )
    assert value["src"] == f"{ASSETS_BASE_URL}/{MESH_ASSET_PATH}/mesh.glb"
    assert value["size_bytes"] == len(mesh_bytes)
    stored_mesh_bytes = Path(get_stored_asset_path(storage_client_with_url_preparator, value["src"])).read_bytes()
    assert stored_mesh_bytes == mesh_bytes, "the mesh file is stored as-is"
    # the stored file is the one the viewers fetch to draw the 3D scene: it must be readable
    reloaded = trimesh.load(BytesIO(stored_mesh_bytes), file_type="glb")
    assert isinstance(reloaded, trimesh.Scene)
    assert len(reloaded.geometry) > 1
    assert_thumbnail(value["thumbnail"], storage_client_with_url_preparator, filename="mesh.glb")


def test_render_trimesh_scene_to_png_without_alpha_channel() -> None:
    """A platform which doesn't render the alpha channel (e.g OSMesa) still gives a valid RGBA image."""
    # the renderer is replaced by a mock which returns what such a platform returns: an RGB image
    rgb_image = np.zeros((MESH_THUMBNAIL_HEIGHT, MESH_THUMBNAIL_WIDTH, 3), dtype=np.uint8)
    with patch("pyrender.OffscreenRenderer") as OffscreenRenderer:
        OffscreenRenderer.return_value.render.return_value = (rgb_image, None)
        output_buffer = BytesIO()
        _render_trimesh_scene_to_png(trimesh.creation.box().scene(), output_buffer)
    output_buffer.seek(0)
    image = PILImage.open(output_buffer)
    assert image.format == "PNG"
    assert image.size == (MESH_THUMBNAIL_WIDTH, MESH_THUMBNAIL_HEIGHT)
    pixel = image.getpixel((0, 0))
    assert isinstance(pixel, tuple)
    assert pixel[3] == 255, "the background is opaque, but the colors are kept"


@pytest.mark.parametrize(
    "texture,expected",
    [
        # pyrender refuses the boolean textures: we convert them to a black and white image
        (
            np.array([[True, False]]),
            np.array([[[255, 255, 255, 255], [0, 0, 0, 255]]], dtype=np.uint8),
        ),
        # pyrender refuses to convert 2-channels textures to RGBA: the missing channel is filled
        (
            np.array([[[10, 20], [30, 40]]], dtype=np.uint8),
            np.array([[[10, 20, 15, 255], [30, 40, 35, 255]]], dtype=np.uint8),
        ),
    ],
)
def test_robust_format_texture_source_fixes(texture: np.ndarray, expected: np.ndarray) -> None:
    from pyrender.utils import format_texture_source  # type: ignore

    out = _robust_format_texture_source(texture, "RGBA")
    assert out is not None
    assert np.array_equal(out, expected)
    # the two cases above are the reason of the existence of _robust_format_texture_source
    with pytest.raises(Exception):
        format_texture_source(texture, "RGBA")


@pytest.mark.parametrize(
    "texture",
    [
        None,
        np.array([[[0, 128, 255]]], dtype=np.uint8),
        np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32),
        np.array([[[0, 128, 255, 255]]], dtype=np.uint8),
    ],
)
def test_robust_format_texture_source_is_a_drop_in_replacement(texture: Optional[np.ndarray]) -> None:
    """Out of the fixed cases, it returns the same as the function of pyrender it replaces."""
    from pyrender.utils import format_texture_source

    out = _robust_format_texture_source(texture, "RGBA")
    if texture is None:
        assert out is None
    else:
        assert out is not None
        assert np.array_equal(
            out,
            format_texture_source(texture, "RGBA"),
        )


def test_robust_format_texture_source_with_pil_image() -> None:
    texture = PILImage.new("P", (2, 2), color=1)
    texture.putpalette([0, 0, 0, 255, 0, 0])
    formatted = _robust_format_texture_source(texture, "RGBA")
    assert formatted is not None
    assert formatted.shape == (2, 2, 4)
    assert formatted[0, 0].tolist() == [255, 0, 0, 255]


def test_robust_format_texture_source_with_unsupported_dtype() -> None:
    with pytest.raises(TypeError):
        _robust_format_texture_source(np.array([[1 + 1j]], dtype=np.complex128), "RGBA")
