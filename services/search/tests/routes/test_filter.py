# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import shutil
from collections.abc import Generator
from http import HTTPStatus
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from fsspec import AbstractFileSystem
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath

from search.config import AppConfig
from search.routes.filter import (
    create_response,
    execute_filter_query,
    execute_filter_query_on_index,
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


@pytest.fixture  # TODO: session scope
def index_file_location(ds: Dataset) -> Generator[str, None, None]:
    index_file_location = "index.duckdb"
    con = duckdb.connect(index_file_location)
    con.execute("INSTALL 'httpfs';")
    con.execute("LOAD 'httpfs';")
    con.execute("INSTALL 'fts';")
    con.execute("LOAD 'fts';")
    con.sql("CREATE OR REPLACE SEQUENCE serial START 0 MINVALUE 0;")
    sample_df = ds.to_pandas()  # noqa: F841
    create_command_sql = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS __hf_index_id, * FROM sample_df"
    con.sql(create_command_sql)
    # assert sample_df.shape[0] == con.execute(query="SELECT COUNT(*) FROM data;").fetchall()[0][0]
    con.sql("PRAGMA create_fts_index('data', '__hf_index_id', 'name', 'gender', overwrite=1);")
    con.close()
    yield index_file_location
    os.remove(index_file_location)


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


def test_execute_filter_query(ds_fs: AbstractFileSystem) -> None:
    parquet_file_paths = [str(ds_fs.tmp_dir.joinpath(path)) for path in ds_fs.ls("default", detail=False)]
    columns, where, limit, offset = ["name", "age"], "gender = 'female'", 1, 1
    num_rows_total, pa_table = execute_filter_query(
        columns=columns, parquet_file_urls=parquet_file_paths, where=where, limit=limit, offset=offset
    )
    assert num_rows_total == 2
    assert pa_table == pa.Table.from_pydict({"name": ["Simone"], "age": [30]})


def test_execute_filter_query_on_index(index_file_location: str) -> None:
    columns, where, limit, offset = ["name", "age"], "gender = 'female'", 1, 1
    num_rows_total, pa_table = execute_filter_query_on_index(
        index_file_location=index_file_location, columns=columns, where=where, limit=limit, offset=offset
    )
    assert num_rows_total == 2
    assert pa_table == pa.Table.from_pydict({"name": ["Simone"], "age": [30]})


def test_create_response(ds: Dataset, app_config: AppConfig, cached_assets_directory: StrPath) -> None:
    dataset, config, split = "ds", "default", "train"
    offset = 2
    pa_table = pa.Table.from_pydict(
        {
            "name": ["Marie", "Paul", "Leo", "Simone"],
            "gender": ["female", "male", "male", "female"],
            "age": [35, 30, 25, 30],
        }
    )
    response = create_response(
        dataset=dataset,
        config=config,
        split=split,
        cached_assets_base_url=app_config.cached_assets.base_url,
        cached_assets_directory=cached_assets_directory,
        pa_table=pa_table,
        offset=offset,
        features=ds.features,
        unsupported_columns=[],
        num_rows_total=4,
    )
    assert response == {
        "features": [
            {"feature_idx": 0, "name": "name", "type": {"dtype": "string", "_type": "Value"}},
            {"feature_idx": 1, "name": "gender", "type": {"dtype": "string", "_type": "Value"}},
            {"feature_idx": 2, "name": "age", "type": {"dtype": "int64", "_type": "Value"}},
        ],
        "rows": [
            {"row_idx": 2, "row": {"name": "Marie", "gender": "female", "age": 35}, "truncated_cells": []},
            {"row_idx": 3, "row": {"name": "Paul", "gender": "male", "age": 30}, "truncated_cells": []},
            {"row_idx": 4, "row": {"name": "Leo", "gender": "male", "age": 25}, "truncated_cells": []},
            {"row_idx": 5, "row": {"name": "Simone", "gender": "female", "age": 30}, "truncated_cells": []},
        ],
        "num_rows_total": 4,
        "num_rows_per_page": 100,
    }
