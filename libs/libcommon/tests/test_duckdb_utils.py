from typing import Any
from unittest.mock import patch

import duckdb
import pytest
from datasets import Dataset

from libcommon.duckdb_utils import (
    DUCKDB_BUILD_MEMORY_LIMIT,
    DUCKDB_BUILD_THREADS,
    DUCKDB_TOKENIZE_MEMORY_LIMIT,
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


def test_create_index_limits_duckdb_resources(tmp_path_factory: pytest.TempPathFactory) -> None:
    database = str(tmp_path_factory.mktemp("index") / "index.duckdb")
    duckdb.execute("INSTALL 'fts';")
    with duckdb_connect(database=database) as con:
        con.sql("CREATE TABLE data AS SELECT range AS __hf_index_id, 'hello world ' || range AS text FROM range(100);")
        con.sql("CHECKPOINT;")
    settings: list[tuple[int, str, str]] = []

    def recording_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        con = duckdb_connect(*args, **kwargs)
        row = con.sql(
            "SELECT current_setting('threads'), current_setting('memory_limit'), current_setting('temp_directory')"
        ).fetchone()
        settings.append(row)  # type: ignore
        return con

    with patch("libcommon.duckdb_utils.duckdb_connect", recording_connect):
        create_index(database, "data", ["text"], "porter", "__hf_index_id", "fts_main_data")

    def memory_limit(value: str) -> str:
        row = duckdb.connect(config={"memory_limit": value}).sql("SELECT current_setting('memory_limit')").fetchone()
        assert row is not None
        return str(row[0])

    # the main connection, the parallel tokenize connections and the merge connection
    build, *tokenize, merge = settings
    assert tokenize
    for threads, limit, _ in (build, merge):
        assert (threads, limit) == (DUCKDB_BUILD_THREADS, memory_limit(DUCKDB_BUILD_MEMORY_LIMIT))
    for threads, limit, _ in tokenize:
        assert (threads, limit) == (DUCKDB_TOKENIZE_THREADS, memory_limit(DUCKDB_TOKENIZE_MEMORY_LIMIT))
    # one spill directory per connection: DuckDB names its temporary files the same way in every connection
    assert len({temp_directory for *_, temp_directory in tokenize}) == len(tokenize)
    assert build[2] == merge[2] not in {temp_directory for *_, temp_directory in tokenize}
