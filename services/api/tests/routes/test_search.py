# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from libcommon.storage import StrPath

from api.routes.search import full_text_search, get_index_folder


def test_get_index_folder(duckdb_index_cache_directory: StrPath) -> None:
    dataset, config, split = "dataset", "config", "split"
    index_folder = get_index_folder(duckdb_index_cache_directory, dataset, config, split)
    assert index_folder is not None
    assert duckdb_index_cache_directory in index_folder


@pytest.mark.parametrize(
    "query,offset,length,expected_result, expected_num_total_items",
    [
        (
            "Lord Vader",
            0,
            100,
            {
                "text": [
                    "Grand Moff Tarkin and Lord Vader are interrupted in their discussion by the buzz of the comlink",
                    "Vader turns round and round in circles as his ship spins into space.",
                    "The wingman spots the pirateship coming at him and warns the Dark Lord",
                ]
            },
            3,
        ),
        (
            "Lord Vader",
            1,
            2,
            {
                "text": [
                    "Vader turns round and round in circles as his ship spins into space.",
                    "The wingman spots the pirateship coming at him and warns the Dark Lord",
                ]
            },
            3,
        ),
        ("non existing text", 0, 100, {"text": []}, 0),
    ],
)
def test_full_text_search(
    query: str, offset: int, length: int, expected_result: Any, expected_num_total_items: int
) -> None:
    index_file_location = "index.duckdb"
    con = duckdb.connect(index_file_location)

    con.execute("INSTALL 'httpfs';")
    con.execute("LOAD 'httpfs';")
    con.execute("INSTALL 'fts';")
    con.execute("LOAD 'fts';")

    con.sql("CREATE OR REPLACE SEQUENCE serial START 1;")

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

    create_command_sql = "CREATE OR REPLACE TABLE data AS SELECT nextval('serial') AS __hf_index_id, * FROM sample_df"
    con.sql(create_command_sql)
    con.execute(query="SELECT COUNT(*) FROM data;").fetchall()
    assert sample_df.size == con.execute(query="SELECT COUNT(*) FROM data;").fetchall()[0][0]
    con.sql("PRAGMA create_fts_index('data', '__hf_index_id', '*', overwrite=1);")
    con.close()

    (num_total_items, pa_table) = full_text_search(index_file_location, query, offset, length)
    assert num_total_items is not None
    assert pa_table is not None
    assert num_total_items == expected_num_total_items

    fields = [pa.field("text", pa.string())]
    filtered_df = pd.DataFrame(expected_result)
    expected_table = pa.Table.from_pandas(filtered_df, schema=pa.schema(fields), preserve_index=False)
    assert pa_table == expected_table

    os.remove(index_file_location)
