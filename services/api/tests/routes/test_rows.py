from http import HTTPStatus
from typing import Any, Generator, List
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset, concatenate_datasets
from fsspec import AbstractFileSystem
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import _clean_cache_database, upsert_response

from api.config import AppConfig
from api.routes.endpoint import StepsByInputTypeAndEndpoint
from api.routes.rows import Indexer


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict({"text": ["Hello there", "General Kenobi"]})


@pytest.fixture
def ds_fs(dataset: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("plain_text/ds-train.parquet", "wb") as f:
        dataset.to_parquet(f)
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
def dataset_with_config_parquet() -> dict[str, Any]:
    config_parquet_content = {
        "parquet_files": [
            {
                "dataset": "ds",
                "config": "plain_text",
                "split": "train",
                "url": "https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/plain_text/ds-train.parquet",
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
                "url": f"https://fake.huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/plain_text/ds_sharded-train-{shard_idx:05d}-of-{num_shards:05d}.parquet",
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
def config_parquet_processing_steps(endpoint_definition: StepsByInputTypeAndEndpoint) -> List[ProcessingStep]:
    parquet_processing_steps_by_input_type = endpoint_definition.get("/parquet")
    if not parquet_processing_steps_by_input_type or not parquet_processing_steps_by_input_type["config"]:
        raise RuntimeError("The parquet endpoint is not configured. Exiting.")
    return parquet_processing_steps_by_input_type["config"]


@pytest.fixture
def indexer(app_config: AppConfig, config_parquet_processing_steps: List[ProcessingStep]) -> Indexer:
    return Indexer(
        config_parquet_processing_steps=config_parquet_processing_steps,
        init_processing_steps=[],
        hf_endpoint=app_config.common.hf_endpoint,
        hf_token=app_config.common.hf_token,
    )


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
