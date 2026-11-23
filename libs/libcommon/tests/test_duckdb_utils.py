from typing import Any
from unittest.mock import patch

import duckdb
import pytest
from datasets import Dataset

from libcommon.duckdb_utils import (
    DUCKDB_BUILD_CONFIG,
    DUCKDB_TOKENIZE_CONFIG,
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
    configs: list[dict[str, Any]] = []

    def recording_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        configs.append(kwargs.get("config", {}))
        return duckdb_connect(*args, **kwargs)  # duckdb raises if a setting is invalid

    with patch("libcommon.duckdb_utils.duckdb_connect", recording_connect):
        create_index(database, "data", ["text"], "porter", "__hf_index_id", "fts_main_data")

    # the main connection, the parallel tokenize connections and the merge connection
    build, *tokenize, merge = [dict(config) for config in configs]
    spill_directories = [config.pop("temp_directory") for config in (build, *tokenize, merge)]
    assert build == merge == DUCKDB_BUILD_CONFIG
    assert tokenize and all(config == DUCKDB_TOKENIZE_CONFIG for config in tokenize)
    # one spill directory per connection: DuckDB names its temporary files the same way in every connection
    assert spill_directories[0] == spill_directories[-1]
    assert len(set(spill_directories)) == len(tokenize) + 1
