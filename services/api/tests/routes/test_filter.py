"""Test filter."""
import shutil
from http import HTTPStatus
from pathlib import Path
from typing import Any, Generator

import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from fsspec import AbstractFileSystem
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath

from api.routes.filter import (
    get_config_parquet_metadata_from_cache,
    get_features_from_parquet_file_metadata,
)


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict(
        {
            "name": ["Marie", "Paul", "Leo", "Simone"],
            "gender": ["female", "male", "male", "female"],
            "age": [35, 30, 25, 30],
        }
    )


@pytest.fixture
def ds_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("default/ds-train.parquet", "wb") as f:
        ds.to_parquet(f)
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
def ds_config_parquet_metadata(ds_fs: AbstractFileSystem, ds_parquet_metadata_dir: StrPath) -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files_metadata": [
            {
                "dataset": "ds",
                "config": "default",
                "split": "train",
                "url": (  # noqa: E501
                    "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/ds-train.parquet"
                ),
                "filename": "ds-train.parquet",
                "size": ds_fs.info("default/ds-train.parquet")["size"],
                "num_rows": pq.read_metadata(ds_fs.open("default/ds-train.parquet")).num_rows,
                "parquet_metadata_subpath": "ds/--/default/ds-train.parquet",
            }
        ]
    }
    upsert_response(
        kind="config-parquet-metadata",
        dataset="ds",
        config="default",
        content=config_parquet_content,
        http_status=HTTPStatus.OK,
        progress=1.0,
    )
    return config_parquet_content


def test_get_config_parquet_metadata_from_cache(
    ds_config_parquet_metadata: dict[str, Any], processing_graph: ProcessingGraph
) -> None:
    dataset, config, split = "ds", "default", "train"
    parquet_file_metadata_items, revision = get_config_parquet_metadata_from_cache(
        dataset=dataset, config=config, split=split, processing_graph=processing_graph
    )
    assert parquet_file_metadata_items == ds_config_parquet_metadata["parquet_files_metadata"]


def test_get_features_from_parquet_file_metadata(
    ds: Dataset, ds_config_parquet_metadata: dict[str, Any], ds_parquet_metadata_dir: StrPath
) -> None:
    parquet_file_metadata_item = ds_config_parquet_metadata["parquet_files_metadata"][0]
    features = get_features_from_parquet_file_metadata(
        parquet_file_metadata_item=parquet_file_metadata_item, parquet_metadata_directory=ds_parquet_metadata_dir
    )
    assert features.to_dict() == {
        "name": {"dtype": "string", "_type": "Value"},
        "gender": {"dtype": "string", "_type": "Value"},
        "age": {"dtype": "int64", "_type": "Value"},
    }
