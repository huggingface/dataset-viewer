import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Generator, List
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset, Image, concatenate_datasets
from datasets.table import embed_table_storage
from fsspec import AbstractFileSystem
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import _clean_cache_database, upsert_response
from libcommon.storage import StrPath
from libcommon.viewer_utils.asset import update_last_modified_date_of_rows_in_assets_dir

from api.config import AppConfig
from api.routes.rows import Indexer, RowsIndex, clean_cached_assets, create_response


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict({"text": ["Hello there", "General Kenobi"]})


@pytest.fixture
def ds_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("plain_text/ds-train.parquet", "wb") as f:
        ds.to_parquet(f)
    yield tmpfs


@pytest.fixture
def ds_sharded(ds: Dataset) -> Dataset:
    return concatenate_datasets([ds] * 4)


@pytest.fixture
def ds_sharded_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    num_shards = 4
    for shard_idx in range(num_shards):
        with tmpfs.open(f"plain_text/ds_sharded-train-{shard_idx:05d}-of-{num_shards:05d}.parquet", "wb") as f:
            ds.to_parquet(f)
    yield tmpfs


@pytest.fixture
def ds_image(image_path: str) -> Dataset:
    ds = Dataset.from_dict({"image": [image_path]}).cast_column("image", Image())
    return Dataset(embed_table_storage(ds.data))


@pytest.fixture
def ds_image_fs(ds_image: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("plain_text/ds_image-train.parquet", "wb") as f:
        ds_image.to_parquet(f)
    yield tmpfs


@pytest.fixture
def dataset_with_config_parquet() -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds",
                "config": "plain_text",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/plain_text/ds-train.parquet",  # noqa: E501
                "filename": "ds-train.parquet",
                "size": 128,
            }
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds",
        config="plain_text",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def dataset_sharded_with_config_parquet() -> dict[str, Any]:
    num_shards = 4
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds_sharded",
                "config": "plain_text",
                "split": "train",
                "url": f"https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/plain_text/ds_sharded-train-{shard_idx:05d}-of-{num_shards:05d}.parquet",  # noqa: E501
                "filename": f"ds_sharded-train-{shard_idx:05d}-of-{num_shards:05d}.parquet",
                "size": 128,
            }
            for shard_idx in range(num_shards)
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds_sharded",
        config="plain_text",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def dataset_image_with_config_parquet() -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds_image",
                "config": "plain_text",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/plain_text/ds_image-train.parquet",  # noqa: E501
                "filename": "ds_image-train.parquet",
                "size": 11128,
            }
        ]
    }
    upsert_response(
        kind="config-parquet",
        dataset="ds_image",
        config="plain_text",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


@pytest.fixture
def indexer(app_config: AppConfig, processing_graph: ProcessingGraph) -> Indexer:
    return Indexer(
        processing_graph=processing_graph,
        hf_endpoint=app_config.common.hf_endpoint,
        hf_token=app_config.common.hf_token,
    )


@pytest.fixture
def rows_index(
    indexer: Indexer,
    ds_sharded: Dataset,
    ds_sharded_fs: AbstractFileSystem,
    dataset_sharded_with_config_parquet: dict[str, Any],
) -> Generator[RowsIndex, None, None]:
    with patch("api.routes.rows.get_parquet_fs", return_value=ds_sharded_fs):
        yield indexer.get_rows_index("ds_sharded", "plain_text", "train")


def test_indexer_get_rows_index(
    indexer: Indexer, ds: Dataset, ds_fs: AbstractFileSystem, dataset_with_config_parquet: dict[str, Any]
) -> None:
    with patch("api.routes.rows.get_parquet_fs", return_value=ds_fs):
        index = indexer.get_rows_index("ds", "plain_text", "train")
    assert index.features == ds.features
    assert index.row_group_offsets.tolist() == [len(ds)]
    assert len(index.row_group_readers) == 1
    row_group_reader = index.row_group_readers[0]
    pa_table = row_group_reader()
    assert pa_table.to_pydict() == ds.to_dict()


def test_indexer_get_rows_index_sharded(
    indexer: Indexer,
    ds: Dataset,
    ds_sharded: Dataset,
    ds_sharded_fs: AbstractFileSystem,
    dataset_sharded_with_config_parquet: dict[str, Any],
) -> None:
    with patch("api.routes.rows.get_parquet_fs", return_value=ds_sharded_fs):
        index = indexer.get_rows_index("ds_sharded", "plain_text", "train")
    assert index.features == ds_sharded.features
    assert index.row_group_offsets.tolist() == np.cumsum([len(ds)] * 4).tolist()
    assert len(index.row_group_readers) == 4
    row_group_reader = index.row_group_readers[0]
    pa_table = row_group_reader()
    assert pa_table.to_pydict() == ds.to_dict()


def test_rows_index_query(rows_index: RowsIndex, ds_sharded: Dataset) -> None:
    assert rows_index.query(offset=1, length=3).to_pydict() == ds_sharded[1:4]
    assert rows_index.query(offset=1, length=-1).to_pydict() == ds_sharded[:0]
    assert rows_index.query(offset=1, length=0).to_pydict() == ds_sharded[:0]
    assert rows_index.query(offset=999999, length=1).to_pydict() == ds_sharded[:0]
    assert rows_index.query(offset=1, length=99999999).to_pydict() == ds_sharded[1:]
    with pytest.raises(IndexError):
        rows_index.query(offset=-1, length=2)


def test_create_response(ds: Dataset, app_config: AppConfig, cached_assets_directory: StrPath) -> None:
    response = create_response(
        dataset="ds",
        config="plain_text",
        split="train",
        cached_assets_base_url=app_config.cached_assets.base_url,
        cached_assets_directory=cached_assets_directory,
        pa_table=ds.data,
        offset=0,
        features=ds.features,
        unsupported_columns=[],
    )
    assert response["features"] == [{"feature_idx": 0, "name": "text", "type": {"dtype": "string", "_type": "Value"}}]
    assert response["rows"] == [
        {"row_idx": 0, "row": {"text": "Hello there"}, "truncated_cells": []},
        {"row_idx": 1, "row": {"text": "General Kenobi"}, "truncated_cells": []},
    ]


def test_create_response_with_image(
    ds_image: Dataset, app_config: AppConfig, cached_assets_directory: StrPath
) -> None:
    response = create_response(
        dataset="ds_image",
        config="plain_text",
        split="train",
        cached_assets_base_url=app_config.cached_assets.base_url,
        cached_assets_directory=cached_assets_directory,
        pa_table=ds_image.data,
        offset=0,
        features=ds_image.features,
        unsupported_columns=[],
    )
    assert response["features"] == [{"feature_idx": 0, "name": "image", "type": {"_type": "Image"}}]
    assert response["rows"] == [
        {
            "row_idx": 0,
            "row": {
                "image": {
                    "src": "http://localhost/cached-assets/ds_image/--/plain_text/train/0/image/image.jpg",
                    "height": 480,
                    "width": 640,
                }
            },
            "truncated_cells": [],
        }
    ]
    cached_image_path = Path(cached_assets_directory) / "ds_image/--/plain_text/train/0/image/image.jpg"
    assert cached_image_path.is_file()


@pytest.mark.parametrize(
    "n_rows,keep_most_recent_rows_number,keep_first_rows_number,max_cleaned_rows_number,expected_remaining_rows",
    [
        (8, 1, 1, 100, [0, 7]),
        (8, 2, 2, 100, [0, 1, 6, 7]),
        (8, 1, 1, 3, [0, 4, 5, 6, 7]),
    ],
)
def test_clean_cached_assets(
    tmp_path: Path,
    n_rows: int,
    keep_most_recent_rows_number: int,
    keep_first_rows_number: int,
    max_cleaned_rows_number: int,
    expected_remaining_rows: list[int],
) -> None:
    cached_assets_directory = tmp_path / "cached-assets"
    split_dir = cached_assets_directory / "ds/--/plain_text/train"
    split_dir.mkdir(parents=True)
    for i in range(n_rows):
        (split_dir / str(i)).mkdir()
        time.sleep(0.01)

    def deterministic_glob_rows_in_assets_dir(
        dataset: str,
        assets_directory: StrPath,
    ) -> List[Path]:
        return sorted(
            list(Path(assets_directory).resolve().glob(os.path.join(dataset, "--", "*", "*", "*"))),
            key=lambda p: int(p.name),
        )

    with patch("api.routes.rows.glob_rows_in_assets_dir", deterministic_glob_rows_in_assets_dir):
        clean_cached_assets(
            "ds",
            cached_assets_directory,
            keep_most_recent_rows_number=keep_most_recent_rows_number,
            keep_first_rows_number=keep_first_rows_number,
            max_cleaned_rows_number=max_cleaned_rows_number,
        )
    remaining_rows = sorted(int(row_dir.name) for row_dir in split_dir.glob("*"))
    assert remaining_rows == expected_remaining_rows


def test_update_last_modified_date_of_rows_in_assets_dir(tmp_path: Path) -> None:
    cached_assets_directory = tmp_path / "cached-assets"
    split_dir = cached_assets_directory / "ds/--/plain_text/train"
    split_dir.mkdir(parents=True)
    n_rows = 8
    for i in range(n_rows):
        (split_dir / str(i)).mkdir()
        time.sleep(0.01)
    update_last_modified_date_of_rows_in_assets_dir(
        dataset="ds",
        config="plain_text",
        split="train",
        offset=2,
        length=3,
        assets_directory=cached_assets_directory,
    )
    most_recent_rows_dirs = sorted(list(split_dir.glob("*")), key=os.path.getmtime, reverse=True)
    most_recent_rows = [int(row_dir.name) for row_dir in most_recent_rows_dirs]
    assert sorted(most_recent_rows[:3]) == [2, 3, 4]
    assert most_recent_rows[3:] == [7, 6, 5, 1, 0]
