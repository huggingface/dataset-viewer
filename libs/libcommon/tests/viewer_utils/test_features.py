# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
import re
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch
from urllib import parse

import boto3
import pytest
import trimesh
from aiobotocore.response import StreamingBody
from datasets import Mesh
from datasets.table import embed_table_storage
from moto import mock_aws
from urllib3._collections import HTTPHeaderDict

from libcommon.config import S3Config
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import URLPreparator
from libcommon.viewer_utils.asset import MESH_THUMBNAIL_HEIGHT, MESH_THUMBNAIL_WIDTH
from libcommon.viewer_utils.features import (
    MESH_FILE_TYPE_ERROR,
    _infer_mesh_file_type,
    get_cell_value,
    infer_audio_file_extension,
    to_features_list,
)

from ..constants import (
    ASSETS_BASE_URL,
    CI_HUB_ENDPOINT,
    DATASETS_NAMES,
    DEFAULT_COLUMN_NAME,
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    DEFAULT_ROW_IDX,
    DEFAULT_SPLIT,
)
from ..mesh_rendering import ensure_mesh_rendering
from ..types import DatasetFixture


def assert_output_has_valid_files(value: Any, storage_client: StorageClient) -> None:
    if isinstance(value, list):
        for item in value:
            assert_output_has_valid_files(item, storage_client=storage_client)
    elif isinstance(value, dict):
        if "src" in value and isinstance(value["src"], str) and value["src"].startswith(storage_client.base_url):
            path = Path(
                storage_client.get_full_path(
                    value["src"][len(storage_client.base_url) + 1 :],  # noqa: E203
                )
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_get_cell_value_value(
    storage_client_with_url_preparator: StorageClient,
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset
    feature = dataset.features[DEFAULT_COLUMN_NAME]
    dataset = dataset.with_format("arrow").map(embed_table_storage)
    cell = dataset[DEFAULT_ROW_IDX].to_pydict()[DEFAULT_COLUMN_NAME][DEFAULT_ROW_IDX]

    expected_cell = dataset_fixture.expected_cell
    value = get_cell_value(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        row_idx=DEFAULT_ROW_IDX,
        cell=cell,
        featureName=DEFAULT_COLUMN_NAME,
        fieldType=feature,
        storage_client=storage_client_with_url_preparator,
        hf_endpoint=CI_HUB_ENDPOINT,
        hf_token=None,
    )
    assert value == expected_cell
    assert_output_has_valid_files(expected_cell, storage_client=storage_client_with_url_preparator)


to_features_list


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_to_features_list(
    datasets_fixtures: Mapping[str, DatasetFixture],
    dataset_name: str,
) -> None:
    datasets_fixture = datasets_fixtures[dataset_name]
    dataset = datasets_fixture.dataset
    value = to_features_list(dataset.features)
    assert len(value) == 1
    first_feature = value[0]
    assert first_feature["feature_idx"] == 0
    assert first_feature["name"] == DEFAULT_COLUMN_NAME
    assert first_feature["type"] == datasets_fixture.expected_feature_type


# specific test created for https://github.com/huggingface/dataset-viewer/issues/2045
# which is reproduced only when using s3 for fsspec
# skipped bc of `AttributeError: 'coroutine' object has no attribute 'items'` during s3fs `_error_wrapper`
@pytest.mark.skip
def test_ogg_audio_with_s3(
    datasets_fixtures: Mapping[str, DatasetFixture],
) -> None:
    dataset_name = "audio_ogg"
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset
    dataset = dataset.with_format("arrow").map(embed_table_storage)
    feature = dataset.features[DEFAULT_COLUMN_NAME]
    bucket_name = "bucket"
    with mock_aws():
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=bucket_name)

        # patch _validate to avoid calling self._fs.ls because of known issue in aiotbotocore
        # at https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L47
        with patch("libcommon.storage_client.StorageClient._validate", return_value=None):
            storage_client = StorageClient(
                protocol="s3",
                storage_root=f"{bucket_name}/not-important",
                base_url=ASSETS_BASE_URL,
                overwrite=True,
                s3_config=S3Config(
                    access_key_id="fake_access_key_id",
                    secret_access_key="fake_secret_access_key",
                    region_name="us-east-1",
                ),
                url_preparator=URLPreparator(
                    url_signer=None, hf_endpoint=CI_HUB_ENDPOINT, assets_base_url=ASSETS_BASE_URL
                ),
            )

        # patch aiobotocore.endpoint.convert_to_response_dict  because of known issue in aiotbotocore
        # at https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L47
        # see https://github.com/getmoto/moto/issues/6836 and https://github.com/aio-libs/aiobotocore/issues/755
        # copied from https://github.com/aio-libs/aiobotocore/blob/master/aiobotocore/endpoint.py#L23
        async def convert_to_response_dict(http_response, operation_model):  # type: ignore
            response_dict = {
                "headers": HTTPHeaderDict({}),
                "status_code": http_response.status_code,
                "context": {
                    "operation_name": operation_model.name,
                },
            }
            if response_dict["status_code"] >= 300:
                response_dict["body"] = await http_response.content
            elif operation_model.has_event_stream_output:
                response_dict["body"] = http_response.raw
            elif operation_model.has_streaming_output:
                length = response_dict["headers"].get("content-length")
                response_dict["body"] = StreamingBody(http_response.raw, length)
            else:
                response_dict["body"] = http_response.content
            return response_dict

        with patch("aiobotocore.endpoint.convert_to_response_dict", side_effect=convert_to_response_dict):
            value = get_cell_value(
                dataset=dataset_name,
                revision=DEFAULT_REVISION,
                config=DEFAULT_CONFIG,
                split=DEFAULT_SPLIT,
                row_idx=DEFAULT_ROW_IDX,
                cell=dataset[DEFAULT_ROW_IDX][DEFAULT_COLUMN_NAME].to_pylist()[0],
                featureName=DEFAULT_COLUMN_NAME,
                fieldType=feature,
                storage_client=storage_client,
                hf_endpoint=CI_HUB_ENDPOINT,
                hf_token=None,
            )
            assert value == dataset_fixture.expected_cell


@pytest.mark.parametrize(
    "audio_file_name,expected_audio_file_extension",
    [("test_audio_44100.wav", ".wav"), ("test_audio_16000.mp3", ".mp3")],
)
def test_infer_audio_file_extension(
    audio_file_name: str, expected_audio_file_extension: str, shared_datadir: Path
) -> None:
    audio_file_bytes = (shared_datadir / audio_file_name).read_bytes()
    audio_file_extension = infer_audio_file_extension(audio_file_bytes)
    assert audio_file_extension == expected_audio_file_extension


# --- 3D meshes ---

# a scene with four geometries and textured materials, used to render the thumbnails of the tests below
PATH_TO_MESH_GLB = Path(__file__).resolve().parent.parent / "fixtures" / "data" / "test_mesh.glb"
MESH_DATASET = "datasets-owner/datasets-name"


def mesh_asset_path(row_idx: int) -> str:
    """Where the assets of a row of the Mesh column of MESH_DATASET are stored."""
    return (
        f"{parse.quote(MESH_DATASET)}/--/{DEFAULT_REVISION}/--/{DEFAULT_CONFIG}/{DEFAULT_SPLIT}"
        f"/{row_idx}/{DEFAULT_COLUMN_NAME}"
    )


class FakeHfFile(BytesIO):
    """A file of the Hub, as `HfFileSystem` opens it: it knows its size, as the real one does."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.size = len(payload)


def get_mesh_cell_value(
    cell: Any,
    storage_client: StorageClient,
    row_idx: int = DEFAULT_ROW_IDX,
    hf_token: Optional[str] = None,
) -> Any:
    """Prepare a cell of a Mesh column, like the first-rows endpoint does."""
    return get_cell_value(
        dataset=MESH_DATASET,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        row_idx=row_idx,
        cell=cell,
        featureName=DEFAULT_COLUMN_NAME,
        fieldType=Mesh(),
        storage_client=storage_client,
        hf_endpoint=CI_HUB_ENDPOINT,
        hf_token=hf_token,
    )


@pytest.mark.parametrize(
    "path,expected_file_type",
    [
        ("mesh.glb", "glb"),
        ("MESH.GLB", "glb"),
        ("dir/mesh.ply", "ply"),
        ("mesh.stl", "stl"),
        ("mesh.obj", "obj"),
        # the path of a file inside an archive, or of a file of the Hub, is chained with "::"
        ("mesh.glb::archive.zip", "glb"),
        # the query string of an URL is not part of the extension
        ("https://host/mesh.ply?X-Amz-Signature=abcdef", "ply"),
        # the unsupported types, and the missing ones
        (None, None),
        ("mesh", None),
        ("mesh.txt", None),
        ("mesh.fbbx", None),
    ],
    ids=[
        "glb",
        "GLB",
        "ply",
        "stl",
        "obj",
        "file of an archive",
        "url with query",
        "no path",
        "no extension",
        "txt",
        "unsupported extension",
    ],
)
def test_infer_mesh_file_type(path: Optional[str], expected_file_type: Optional[str]) -> None:
    assert _infer_mesh_file_type(path) == expected_file_type


def test_get_mesh_cell_value_with_bytes(storage_client_with_url_preparator: StorageClient) -> None:
    """The mesh bytes of a cell are stored, with a rendered thumbnail of the 3D scene."""
    ensure_mesh_rendering()
    asset_path = mesh_asset_path(1)
    mesh_bytes = trimesh.load(PATH_TO_MESH_GLB).export(file_type="glb")
    value = get_mesh_cell_value({"path": "mesh.glb", "bytes": mesh_bytes}, storage_client_with_url_preparator, 1)
    assert value == {
        "src": f"{ASSETS_BASE_URL}/{asset_path}/mesh.glb",
        "size_bytes": len(mesh_bytes),
        "thumbnail": {
            "src": f"{ASSETS_BASE_URL}/{asset_path}/mesh.glb.png",
            "height": MESH_THUMBNAIL_HEIGHT,
            "width": MESH_THUMBNAIL_WIDTH,
        },
    }
    assert_output_has_valid_files(value, storage_client_with_url_preparator)
    # the mesh file is stored as-is: it's the one the viewers fetch to draw the 3D scene
    with storage_client_with_url_preparator._fs.open(
        storage_client_with_url_preparator.get_full_path(f"{asset_path}/mesh.glb"), "rb"
    ) as file:
        assert file.read() == mesh_bytes


@pytest.mark.parametrize("path", [None, "mesh", "mesh.txt", "mesh.obj.bak"])
def test_get_mesh_cell_value_without_known_file_type(
    path: Optional[str], storage_client_with_url_preparator: StorageClient
) -> None:
    """The file type of the mesh bytes is read from the extension of the path: it's required."""
    with pytest.raises(ValueError, match=re.escape(MESH_FILE_TYPE_ERROR)):
        get_mesh_cell_value({"path": path, "bytes": b"not a mesh"}, storage_client_with_url_preparator)


def test_get_mesh_cell_value_with_bytes_only(storage_client_with_url_preparator: StorageClient) -> None:
    """Same reason: the bytes alone cannot tell which file type they contain."""
    with pytest.raises(ValueError, match=re.escape(MESH_FILE_TYPE_ERROR)):
        get_mesh_cell_value(b"not a mesh", storage_client_with_url_preparator)


def test_get_mesh_cell_value_with_local_path(storage_client_with_url_preparator: StorageClient) -> None:
    """A local path is skipped, as for the other media: the file cannot be fetched by the viewer."""
    cell = {"path": str(Path("path/of/the/mesh.glb")), "bytes": None}
    assert get_mesh_cell_value(cell, storage_client_with_url_preparator) is None


def test_get_mesh_cell_value_with_hf_path(storage_client_with_url_preparator: StorageClient) -> None:
    """A path of the dataset on the Hub is served from the Hub: only the thumbnail is stored."""
    ensure_mesh_rendering()
    asset_path = mesh_asset_path(2)
    mesh_bytes = PATH_TO_MESH_GLB.read_bytes()
    path = f"hf://datasets/{MESH_DATASET}@{DEFAULT_REVISION}/data/mesh.glb"
    with patch("libcommon.viewer_utils.features.HfFileSystem") as HfFileSystem:
        HfFileSystem.return_value.open.return_value = FakeHfFile(mesh_bytes)
        value = get_mesh_cell_value(
            {"path": path, "bytes": None}, storage_client_with_url_preparator, row_idx=2, hf_token="api_token"
        )
        HfFileSystem.assert_called_once_with(endpoint=CI_HUB_ENDPOINT, token="api_token")
    assert value == {
        "src": f"{CI_HUB_ENDPOINT}/datasets/{MESH_DATASET}/resolve/{DEFAULT_REVISION}/data/mesh.glb",
        "size_bytes": len(mesh_bytes),
        "thumbnail": {
            "src": f"{ASSETS_BASE_URL}/{asset_path}/mesh.glb.png",
            "height": MESH_THUMBNAIL_HEIGHT,
            "width": MESH_THUMBNAIL_WIDTH,
        },
    }
    assert_output_has_valid_files(value, storage_client_with_url_preparator)
    # the 3D file is not stored, since the viewer fetches it from the Hub
    assert not storage_client_with_url_preparator.exists(f"{asset_path}/mesh.glb")


def test_get_mesh_cell_value_with_path_of_other_dataset(storage_client_with_url_preparator: StorageClient) -> None:
    """Only the files of the dataset itself can be fetched and served: a path to another one is refused."""
    with patch("libcommon.viewer_utils.features.HfFileSystem") as HfFileSystem:
        with pytest.raises(ValueError, match="invalid path"):
            get_mesh_cell_value(
                {"path": f"hf://datasets/victim/private-dataset@{DEFAULT_REVISION}/mesh.glb", "bytes": None},
                storage_client_with_url_preparator,
            )
        HfFileSystem.assert_not_called()


@pytest.mark.parametrize("cell_type", ["trimesh", "scene"])
def test_get_mesh_cell_value_with_mesh_object(
    cell_type: str, storage_client_with_url_preparator: StorageClient
) -> None:
    """A cell which contains a decoded mesh, and not the bytes of a mesh file, is exported to glTF binary."""
    ensure_mesh_rendering()
    asset_path = mesh_asset_path(3)
    box = trimesh.creation.box()
    value = get_mesh_cell_value(
        box.scene() if cell_type == "scene" else box, storage_client_with_url_preparator, row_idx=3
    )
    assert value["src"] == f"{ASSETS_BASE_URL}/{asset_path}/mesh.glb"
    assert value["size_bytes"] > 0
    assert_output_has_valid_files(value, storage_client_with_url_preparator)
    with storage_client_with_url_preparator._fs.open(
        storage_client_with_url_preparator.get_full_path(f"{asset_path}/mesh.glb"), "rb"
    ) as file:
        assert len(trimesh.load(BytesIO(file.read()), file_type="glb").geometry) == 1, "the exported mesh loads back"


@pytest.mark.parametrize("cell", ["a string", 42, {"path": None, "bytes": None}])
def test_get_mesh_cell_value_with_unsupported_value(
    cell: Any, storage_client_with_url_preparator: StorageClient
) -> None:
    with pytest.raises(TypeError, match="Mesh cell must be a `trimesh.Scene` object"):
        get_mesh_cell_value(cell, storage_client_with_url_preparator)
