# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
from collections.abc import Generator
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datasets import Dataset
from libapi.exceptions import InvalidParameterError
from libapi.response import ROW_IDX_COLUMN, create_response
from libcommon.storage_client import StorageClient

from search.config import AppConfig
from search.routes.filter import execute_filter_query, validate_query_parameter

CACHED_ASSETS_FOLDER = "cached-assets"

pytestmark = pytest.mark.anyio


@pytest.fixture
def storage_client(tmp_path: Path, app_config: AppConfig) -> StorageClient:
    return StorageClient(
        protocol="file",
        storage_root=str(tmp_path / CACHED_ASSETS_FOLDER),
        base_url=app_config.cached_assets.base_url,
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


@pytest.mark.parametrize("parameter_name, parameter_value", [("where", "col='A'")])
def test_validate_query_parameter(parameter_name: str, parameter_value: str) -> None:
    validate_query_parameter(parameter_value, parameter_name)


@pytest.mark.parametrize(
    "parameter_name, parameter_value",
    [("where", "col='A'; SELECT * from data"), ("where", "col='A' /*"), ("where", "col='A'--")],
)
def test_validate_query_parameter_raises(parameter_name: str, parameter_value: str) -> None:
    with pytest.raises(InvalidParameterError):
        validate_query_parameter(parameter_value, parameter_name)


@pytest.mark.parametrize("columns", [["name", "age"], ["name"]])
@pytest.mark.parametrize("orderby", ["", "age", "age DESC"])
def test_execute_filter_query(columns: list[str], orderby: str, index_file_location: str) -> None:
    # in split-duckdb-index we always add the ROW_IDX_COLUMN column
    # see https://github.com/huggingface/dataset-viewer/blob/main/services/worker/src/worker/job_runners/split/duckdb_index.py#L305
    columns = columns + [ROW_IDX_COLUMN]
    where, limit, offset = "gender='female'", 1, 1
    num_rows_total, pa_table = execute_filter_query(
        index_file_location=index_file_location,
        columns=columns,
        where=where,
        orderby=orderby,
        limit=limit,
        offset=offset,
    )
    assert num_rows_total == 2
    expected_pa_table = pa.Table.from_pydict(
        {
            "__hf_index_id": [0, 1, 2, 3],
            "name": ["Marie", "Paul", "Leo", "Simone"],
            "gender": ["female", "male", "male", "female"],
            "age": [35, 30, 25, 30],
        }
    ).filter(pc.field("gender") == "female")
    if orderby:
        if orderby.endswith(" DESC"):
            sorting = [(orderby.removesuffix(" DESC"), "descending")]
            expected_pa_table = expected_pa_table.sort_by(sorting)
        else:
            expected_pa_table = expected_pa_table.sort_by(orderby)
    expected_pa_table = expected_pa_table.slice(offset, limit).select(columns)
    assert pa_table == expected_pa_table


@pytest.mark.parametrize("where", ["non-existing-column=30", "name=30", "name>30"])
def test_execute_filter_query_raises(where: str, index_file_location: str) -> None:
    columns, limit, offset = ["name", "gender", "age"], 100, 0
    with pytest.raises(InvalidParameterError):
        _ = execute_filter_query(
            index_file_location=index_file_location,
            columns=columns,
            where=where,
            orderby="",
            limit=limit,
            offset=offset,
        )


async def test_create_response(ds: Dataset, app_config: AppConfig, storage_client: StorageClient) -> None:
    dataset, config, split = "ds", "default", "train"
    pa_table = pa.Table.from_pydict(
        {
            "__hf_index_id": [2, 3, 4, 5],
            "name": ["Marie", "Paul", "Leo", "Simone"],
            "gender": ["female", "male", "male", "female"],
            "age": [35, 30, 25, 30],
        }
    )
    response = await create_response(
        dataset=dataset,
        revision="revision",
        config=config,
        split=split,
        storage_client=storage_client,
        pa_table=pa_table,
        offset=0,
        features=ds.features,
        unsupported_columns=[],
        num_rows_total=4,
        partial=False,
        use_row_idx_column=True,
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
        "partial": False,
    }
