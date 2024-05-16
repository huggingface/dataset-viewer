# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from libapi.duckdb import get_download_folder
from libcommon.storage import StrPath

from search.routes.search import full_text_search
from libcommon.constants import ROW_IDX_COLUMN 

def test_get_download_folder(duckdb_index_cache_directory: StrPath) -> None:
    dataset, config, split, revision = "dataset", "config", "split", "revision"
    index_folder = get_download_folder(duckdb_index_cache_directory, 100, dataset, config, split, revision)
    assert index_folder is not None
    assert str(duckdb_index_cache_directory) in index_folder


@pytest.mark.parametrize(
    "query,offset,length,expected_result,expected_num_rows_total",
    [
        (
            "Lord Vader",
            0,
            100,
            {
                ROW_IDX_COLUMN: [0, 4, 2],
                "text": [
                    "Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the comlink",
                    "The wingman spots the pirateship coming at him and warns the Dark Lord",
                    "Vader turns round and round in circles as his ship spins into space.",
                ],
            },
            3,
        ),
        (
            "Lord Vader",
            1,
            2,
            {
                ROW_IDX_COLUMN: [4, 2],
                "text": [
                    "The wingman spots the pirateship coming at him and warns the Dark Lord",
                    "Vader turns round and round in circles as his ship spins into space.",
                ],
            },
            3,
        ),
        ("non existing text", 0, 100, {ROW_IDX_COLUMN: [], "text": []}, 0),
        (";DROP TABLE data;", 0, 100, {ROW_IDX_COLUMN: [], "text": []}, 0),
        ("some text'); DROP TABLE data; --", 0, 100, {ROW_IDX_COLUMN: [], "text": []}, 0),
    ],
)
def test_full_text_search(
    query: str, offset: int, length: int, expected_result: Any, expected_num_rows_total: int
) -> None:
    # simulate index file
    index_file_location = "index.duckdb"
    con = duckdb.connect(index_file_location)
    con.execute("INSTALL 'httpfs';")
    con.execute("LOAD 'httpfs';")
    con.execute("INSTALL 'fts';")
    con.execute("LOAD 'fts';")
    con.sql("CREATE OR REPLACE SEQUENCE serial START 0 MINVALUE 0;")
    sample_df = pd.DataFrame(
        {
            "text": [
                "Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the comlink",
                "There goes another one.",
                "Vader turns round and round in circles as his ship spins into space.",
                "We count thirty Rebel ships.",
                "The wingman spots the pirateship coming at him and warns the Dark Lord",
            ]
        },
        dtype=pd.StringDtype(storage="python"),
    )
    create_command_sql = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS {ROW_IDX_COLUMN}, * FROM sample_df"
    con.sql(create_command_sql)
    con.execute(query="SELECT COUNT(*) FROM data;").fetchall()
    assert sample_df.size == con.execute(query="SELECT COUNT(*) FROM data;").fetchall()[0][0]
    con.sql(f"PRAGMA create_fts_index('data', '{ROW_IDX_COLUMN}', '*', overwrite=1);")
    con.close()

    fields = [pa.field(ROW_IDX_COLUMN, pa.int64()), pa.field("text", pa.string())]
    filtered_df = pd.DataFrame(expected_result)
    expected_table = pa.Table.from_pandas(filtered_df, schema=pa.schema(fields), preserve_index=False)

    # assert search results
    (num_rows_total, pa_table) = full_text_search(index_file_location, query, offset, length)
    assert num_rows_total is not None
    assert pa_table is not None
    assert num_rows_total == expected_num_rows_total
    assert pa_table == expected_table

    # ensure that database has not been modified
    con = duckdb.connect(index_file_location)
    assert sample_df.size == con.execute(query="SELECT COUNT(*) FROM data;").fetchall()[0][0]
    con.close()

    os.remove(index_file_location)
