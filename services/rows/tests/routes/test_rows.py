# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import shutil
from collections.abc import Generator
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest
from datasets import Dataset, Image, concatenate_datasets
from datasets.table import embed_table_storage
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from libapi.response import create_response
from libcommon.parquet_utils import (
    Indexer,
    ParquetIndexWithMetadata,
    RowsIndex,
    TooBigRows,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import _clean_cache_database, upsert_response
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from PIL import Image as PILImage  # type: ignore

from rows.config import AppConfig

REVISION_NAME = "revision"
CACHED_ASSETS_FOLDER = "cached-assets"

pytestmark = pytest.mark.anyio


@pytest.fixture
def storage_client(tmp_path: Path) -> StorageClient:
    return StorageClient(
        protocol="file",
        root=str(tmp_path),
        folder=CACHED_ASSETS_FOLDER,
    )


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()


@pytest.fixture(autouse=True)
def enable_parquet_metadata_on_all_datasets() -> Generator[None, None, None]:
    with patch("rows.routes.rows.ALL_COLUMNS_SUPPORTED_DATASETS_ALLOW_LIST", "all"):
        yield


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict({"text": ["Hello there", "General Kenobi"]})


@pytest.fixture
def ds_empty() -> Dataset:
    return Dataset.from_dict({"text": ["Hello there", "General Kenobi"]}).select([])


@pytest.fixture
def ds_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("default/train/0000.parquet", "wb") as f:
        ds.to_parquet(f)
    yield tmpfs


@pytest.fixture
def ds_empty_fs(ds_empty: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("default/train/0000.parquet", "wb") as f:
        ds_empty.to_parquet(f)
    yield tmpfs


@pytest.fixture
def ds_sharded(ds: Dataset) -> Dataset:
    return concatenate_datasets([ds] * 4)


@pytest.fixture
def ds_sharded_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    num_shards = 4
    for shard_idx in range(num_shards):
        with tmpfs.open(f"default/train/{shard_idx:04d}.parquet", "wb") as f:
            ds.to_parquet(f)
    yield tmpfs


@pytest.fixture
def ds_image(image_path: str) -> Dataset:
    ds = Dataset.from_dict({"image": [image_path]}).cast_column("image", Image())
    return Dataset(embed_table_storage(ds.data))


@pytest.fixture
def ds_image_fs(ds_image: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("default/train/0000.parquet", "wb") as f:
        ds_image.to_parquet(f)
    yield tmpfs


@pytest.fixture
def ds_parquet_metadata_dir(
    ds_fs: AbstractFileSystem, parquet_metadata_directory: StrPath
) -> Generator[StrPath, None, None]:
    parquet_shard_paths = ds_fs.glob("**.parquet")
    for parquet_shard_path in parquet_shard_paths:
        parquet_file_metadata_path = Path(parquet_metadata_directory) / "ds" / "--" / parquet_shard_path
        parquet_file_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with ds_fs.open(parquet_shard_path) as parquet_shard_f:
            with open(parquet_file_metadata_path, "wb") as parquet_file_metadata_f:
                pq.read_metadata(parquet_shard_f).write_metadata_file(parquet_file_metadata_f)
    yield parquet_metadata_directory
    shutil.rmtree(Path(parquet_metadata_directory) / "ds")


@pytest.fixture
def dataset_with_config_parquet() -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds",
                "config": "default",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",  # noqa: E501
                "filename": "0000.parquet",
                "size": 128,
            }
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def dataset_with_config_parquet_metadata(
    ds_fs: AbstractFileSystem, ds_parquet_metadata_dir: StrPath
) -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files_metadata": [
            {
                "dataset": "ds",
                "config": "default",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",  # noqa: E501
                "filename": "0000.parquet",
                "size": ds_fs.info("default/train/0000.parquet")["size"],
                "num_rows": pq.read_metadata(ds_fs.open("default/train/0000.parquet")).num_rows,
                "parquet_metadata_subpath": "ds/--/default/train/0000.parquet",
            }
        ]
    }
    upsert_response(
        kind="config-parquet-metadata",
        dataset="ds",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def ds_empty_parquet_metadata_dir(
    ds_empty_fs: AbstractFileSystem, parquet_metadata_directory: StrPath
) -> Generator[StrPath, None, None]:
    parquet_shard_paths = ds_empty_fs.glob("**.parquet")
    for parquet_shard_path in parquet_shard_paths:
        parquet_file_metadata_path = Path(parquet_metadata_directory) / "ds_empty" / "--" / parquet_shard_path
        parquet_file_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with ds_empty_fs.open(parquet_shard_path) as parquet_shard_f:
            with open(parquet_file_metadata_path, "wb") as parquet_file_metadata_f:
                pq.read_metadata(parquet_shard_f).write_metadata_file(parquet_file_metadata_f)
    yield parquet_metadata_directory
    shutil.rmtree(Path(parquet_metadata_directory) / "ds_empty")


@pytest.fixture
def dataset_empty_with_config_parquet() -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds_empty",
                "config": "default",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds_empty/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",  # noqa: E501
                "filename": "0000.parquet",
                "size": 128,
            }
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds_empty",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def dataset_empty_with_config_parquet_metadata(
    ds_empty_fs: AbstractFileSystem, ds_empty_parquet_metadata_dir: StrPath
) -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files_metadata": [
            {
                "dataset": "ds_empty",
                "config": "default",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",  # noqa: E501
                "filename": "0000.parquet",
                "size": ds_empty_fs.info("default/train/0000.parquet")["size"],
                "num_rows": pq.read_metadata(ds_empty_fs.open("default/train/0000.parquet")).num_rows,
                "parquet_metadata_subpath": "ds_empty/--/default/train/0000.parquet",
            }
        ]
    }
    upsert_response(
        kind="config-parquet-metadata",
        dataset="ds_empty",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def ds_sharded_parquet_metadata_dir(
    ds_sharded_fs: AbstractFileSystem, parquet_metadata_directory: StrPath
) -> Generator[StrPath, None, None]:
    parquet_shard_paths = ds_sharded_fs.glob("**.parquet")
    for parquet_shard_path in parquet_shard_paths:
        parquet_file_metadata_path = Path(parquet_metadata_directory) / "ds_sharded" / "--" / parquet_shard_path
        parquet_file_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with ds_sharded_fs.open(parquet_shard_path) as parquet_shard_f:
            with open(parquet_file_metadata_path, "wb") as parquet_file_metadata_f:
                pq.read_metadata(parquet_shard_f).write_metadata_file(parquet_file_metadata_f)
    yield parquet_metadata_directory
    shutil.rmtree(Path(parquet_metadata_directory) / "ds_sharded")


@pytest.fixture
def dataset_sharded_with_config_parquet() -> dict[str, Any]:
    num_shards = 4
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds_sharded",
                "config": "default",
                "split": "train",
                "url": f"https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train{shard_idx:04d}.parquet",  # noqa: E501
                "filename": f"{shard_idx:04d}.parquet",
                "size": 128,
            }
            for shard_idx in range(num_shards)
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds_sharded",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def dataset_sharded_with_config_parquet_metadata(
    ds_sharded_fs: AbstractFileSystem, ds_sharded_parquet_metadata_dir: StrPath
) -> dict[str, Any]:
    config_parquet_metadata_content = {
        "parquet_files_metadata": [
            {
                "dataset": "ds_sharded",
                "config": "default",
                "split": "train",
                "url": f"https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/{parquet_file_path}",  # noqa: E501
                "filename": os.path.basename(parquet_file_path),
                "size": ds_sharded_fs.info(parquet_file_path)["size"],
                "num_rows": pq.read_metadata(ds_sharded_fs.open(parquet_file_path)).num_rows,
                "parquet_metadata_subpath": f"ds_sharded/--/{parquet_file_path}",
            }
            for parquet_file_path in ds_sharded_fs.glob("default/**.parquet")
        ]
    }
    upsert_response(
        kind="config-parquet-metadata",
        dataset="ds_sharded",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_metadata_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_metadata_content


@pytest.fixture
def dataset_image_with_config_parquet() -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds_image",
                "config": "default",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",  # noqa: E501
                "filename": "0000.parquet",
                "size": 11128,
            }
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds_image",
        dataset_git_revision=REVISION_NAME,
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def indexer(
    app_config: AppConfig,
    processing_graph: ProcessingGraph,
    parquet_metadata_directory: StrPath,
) -> Indexer:
    return Indexer(
        processing_graph=processing_graph,
        hf_token=app_config.common.hf_token,
        parquet_metadata_directory=parquet_metadata_directory,
        httpfs=HTTPFileSystem(),
        max_arrow_data_in_memory=9999999999,
    )


@pytest.fixture
def rows_index_with_parquet_metadata(
    indexer: Indexer,
    ds_sharded: Dataset,
    ds_sharded_fs: AbstractFileSystem,
    dataset_sharded_with_config_parquet_metadata: dict[str, Any],
) -> Generator[RowsIndex, None, None]:
    with ds_sharded_fs.open("default/train/0003.parquet") as f:
        with patch("libcommon.parquet_utils.HTTPFile", return_value=f):
            yield indexer.get_rows_index("ds_sharded", "default", "train")


@pytest.fixture
def rows_index_with_empty_dataset(
    indexer: Indexer,
    ds_empty: Dataset,
    ds_empty_fs: AbstractFileSystem,
    dataset_empty_with_config_parquet_metadata: dict[str, Any],
) -> Generator[RowsIndex, None, None]:
    with ds_empty_fs.open("default/train/0000.parquet") as f:
        with patch("libcommon.parquet_utils.HTTPFile", return_value=f):
            yield indexer.get_rows_index("ds_empty", "default", "train")


@pytest.fixture
def rows_index_with_too_big_rows(
    app_config: AppConfig,
    processing_graph: ProcessingGraph,
    parquet_metadata_directory: StrPath,
    ds_sharded: Dataset,
    ds_sharded_fs: AbstractFileSystem,
    dataset_sharded_with_config_parquet_metadata: dict[str, Any],
) -> Generator[RowsIndex, None, None]:
    indexer = Indexer(
        processing_graph=processing_graph,
        hf_token=app_config.common.hf_token,
        parquet_metadata_directory=parquet_metadata_directory,
        httpfs=HTTPFileSystem(),
        max_arrow_data_in_memory=1,
    )
    with ds_sharded_fs.open("default/train/0003.parquet") as f:
        with patch("libcommon.parquet_utils.HTTPFile", return_value=f):
            yield indexer.get_rows_index("ds_sharded", "default", "train")


def test_indexer_get_rows_index_with_parquet_metadata(
    indexer: Indexer, ds: Dataset, ds_fs: AbstractFileSystem, dataset_with_config_parquet_metadata: dict[str, Any]
) -> None:
    with ds_fs.open("default/train/0000.parquet") as f:
        with patch("libcommon.parquet_utils.HTTPFile", return_value=f):
            index = indexer.get_rows_index("ds", "default", "train")
    assert isinstance(index.parquet_index, ParquetIndexWithMetadata)
    assert index.parquet_index.features == ds.features
    assert index.parquet_index.num_rows == [len(ds)]
    assert index.parquet_index.num_rows_total == 2
    assert index.parquet_index.parquet_files_urls == [
        parquet_file_metadata_item["url"]
        for parquet_file_metadata_item in dataset_with_config_parquet_metadata["parquet_files_metadata"]
    ]
    assert len(index.parquet_index.metadata_paths) == 1
    assert os.path.exists(index.parquet_index.metadata_paths[0])


def test_indexer_get_rows_index_sharded_with_parquet_metadata(
    indexer: Indexer,
    ds: Dataset,
    ds_sharded: Dataset,
    ds_sharded_fs: AbstractFileSystem,
    dataset_sharded_with_config_parquet_metadata: dict[str, Any],
) -> None:
    with ds_sharded_fs.open("default/train/0003.parquet") as f:
        with patch("libcommon.parquet_utils.HTTPFile", return_value=f):
            index = indexer.get_rows_index("ds_sharded", "default", "train")
    assert isinstance(index.parquet_index, ParquetIndexWithMetadata)
    assert index.parquet_index.features == ds_sharded.features
    assert index.parquet_index.num_rows == [len(ds)] * 4
    assert index.parquet_index.num_rows_total == 8
    assert index.parquet_index.parquet_files_urls == [
        parquet_file_metadata_item["url"]
        for parquet_file_metadata_item in dataset_sharded_with_config_parquet_metadata["parquet_files_metadata"]
    ]
    assert len(index.parquet_index.metadata_paths) == 4
    assert all(os.path.exists(index.parquet_index.metadata_paths[i]) for i in range(4))


def test_rows_index_query_with_parquet_metadata(
    rows_index_with_parquet_metadata: RowsIndex, ds_sharded: Dataset
) -> None:
    assert isinstance(rows_index_with_parquet_metadata.parquet_index, ParquetIndexWithMetadata)
    assert rows_index_with_parquet_metadata.query(offset=1, length=3).to_pydict() == ds_sharded[1:4]
    assert rows_index_with_parquet_metadata.query(offset=1, length=-1).to_pydict() == ds_sharded[:0]
    assert rows_index_with_parquet_metadata.query(offset=1, length=0).to_pydict() == ds_sharded[:0]
    assert rows_index_with_parquet_metadata.query(offset=999999, length=1).to_pydict() == ds_sharded[:0]
    assert rows_index_with_parquet_metadata.query(offset=1, length=99999999).to_pydict() == ds_sharded[1:]
    with pytest.raises(IndexError):
        rows_index_with_parquet_metadata.query(offset=-1, length=2)


def test_rows_index_query_with_too_big_rows(rows_index_with_too_big_rows: RowsIndex, ds_sharded: Dataset) -> None:
    with pytest.raises(TooBigRows):
        rows_index_with_too_big_rows.query(offset=0, length=3)


def test_rows_index_query_with_empty_dataset(rows_index_with_empty_dataset: RowsIndex, ds_sharded: Dataset) -> None:
    assert isinstance(rows_index_with_empty_dataset.parquet_index, ParquetIndexWithMetadata)
    assert rows_index_with_empty_dataset.query(offset=0, length=1).to_pydict() == ds_sharded[:0]
    with pytest.raises(IndexError):
        rows_index_with_empty_dataset.query(offset=-1, length=2)


async def test_create_response(ds: Dataset, app_config: AppConfig, storage_client: StorageClient) -> None:
    response = await create_response(
        dataset="ds",
        revision="revision",
        config="default",
        split="train",
        cached_assets_base_url=app_config.cached_assets.base_url,
        storage_client=storage_client,
        pa_table=ds.data,
        offset=0,
        features=ds.features,
        unsupported_columns=[],
        num_rows_total=10,
        partial=False,
    )
    assert response["features"] == [{"feature_idx": 0, "name": "text", "type": {"dtype": "string", "_type": "Value"}}]
    assert response["rows"] == [
        {"row_idx": 0, "row": {"text": "Hello there"}, "truncated_cells": []},
        {"row_idx": 1, "row": {"text": "General Kenobi"}, "truncated_cells": []},
    ]
    assert response["num_rows_total"] == 10
    assert response["num_rows_per_page"] == 100
    assert response["partial"] is False


async def test_create_response_with_image(
    ds_image: Dataset, app_config: AppConfig, storage_client: StorageClient
) -> None:
    dataset, config, split = "ds_image", "default", "train"
    image_key = "ds_image/--/revision/--/default/train/0/image/image.jpg"
    response = await create_response(
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        cached_assets_base_url=app_config.cached_assets.base_url,
        storage_client=storage_client,
        pa_table=ds_image.data,
        offset=0,
        features=ds_image.features,
        unsupported_columns=[],
        num_rows_total=10,
        partial=False,
    )
    assert response["features"] == [{"feature_idx": 0, "name": "image", "type": {"_type": "Image"}}]
    assert response["rows"] == [
        {
            "row_idx": 0,
            "row": {
                "image": {
                    "src": f"{app_config.cached_assets.base_url}/{image_key}",
                    "height": 480,
                    "width": 640,
                }
            },
            "truncated_cells": [],
        }
    ]
    assert response["partial"] is False
    assert storage_client.exists(image_key)
    image = PILImage.open(f"{storage_client.get_base_directory()}/{image_key}")
    assert image is not None
