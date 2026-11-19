from typing import Any
from unittest.mock import patch

import duckdb
import pytest
from datasets import Dataset

from libcommon.duckdb_utils import (
    DUCKDB_BUILD_THREADS,
    DUCKDB_TOKENIZE_THREADS,
    compute_length_column,
    create_index,
    duckdb_connect,
)


def test_compute_length_column_for_list_of_json(tmp_path_factory: pytest.TempPathFactory) -> None:
    parquet_directory = tmp_path_factory.mktemp("data")
    parquet_filename = parquet_directory / "list_of_json.parquet"

    data = {"col": [[0, "a"]]}
    ds = Dataset.from_dict(data, on_mixed_types="use_json")
    ds.to_parquet(parquet_filename)

    length_column = compute_length_column([parquet_filename], "col", dtype="list", target_df=None)
    assert length_column.to_dict(as_series=False) == {"col.length": [2]}


def test_create_index_limits_duckdb_threads(tmp_path_factory: pytest.TempPathFactory) -> None:
    database = str(tmp_path_factory.mktemp("index") / "index.duckdb")
    duckdb.execute("INSTALL 'fts';")
    with duckdb_connect(database=database) as con:
        con.sql("CREATE TABLE data AS SELECT range AS __hf_index_id, 'hello world ' || range AS text FROM range(100);")
        con.sql("CHECKPOINT;")
    threads: list[int] = []

    def recording_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        con = duckdb_connect(*args, **kwargs)
        threads.append(con.sql("SELECT current_setting('threads')").fetchone()[0])  # type: ignore
        return con

    with patch("libcommon.duckdb_utils.duckdb_connect", recording_connect):
        create_index(database, "data", ["text"], "porter", "__hf_index_id", "fts_main_data")

    # the main connection, the parallel tokenize connections and the merge connection
    assert threads[0] == threads[-1] == DUCKDB_BUILD_THREADS
    assert set(threads[1:-1]) == {DUCKDB_TOKENIZE_THREADS}
    assert len(threads) > 2
