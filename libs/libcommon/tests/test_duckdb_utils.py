from typing import Any
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from libcommon.duckdb_utils import (
    CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES,
    DUCKDB_BUILD_CONFIG,
    DUCKDB_TOKENIZE_CONFIG,
    compute_length_column,
    create_index,
    duckdb_connect,
    varchar_sql,
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


def test_create_index_runs_out_of_memory_tokenize_jobs_again(tmp_path_factory: pytest.TempPathFactory) -> None:
    database = str(tmp_path_factory.mktemp("index") / "index.duckdb")
    duckdb.execute("INSTALL 'fts';")
    with duckdb_connect(database=database) as con:
        con.sql("CREATE TABLE data AS SELECT range AS __hf_index_id, 'hello world ' || range AS text FROM range(100);")
        con.sql("CHECKPOINT;")
    retried: list[str] = []

    def connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        job_databases = kwargs.get("tmp_subprocess_job_databases") or []
        if len(job_databases) == 1 and job_databases[0].endswith("tmp_0_0.duckdb"):  # not the merge connection
            if kwargs["config"]["memory_limit"] == DUCKDB_TOKENIZE_CONFIG["memory_limit"]:
                raise duckdb.OutOfMemoryException("Out of Memory Error: failed to allocate data")
            retried.append(kwargs["config"]["memory_limit"])
        return duckdb_connect(*args, **kwargs)

    with patch("libcommon.duckdb_utils.duckdb_connect", connect):
        create_index(database, "data", ["text"], "porter", "__hf_index_id", "fts_main_data")

    assert retried == [DUCKDB_BUILD_CONFIG["memory_limit"]]
    with duckdb_connect(database=database, read_only=True) as con:
        con.sql("LOAD fts;")
        hits = con.sql(
            "SELECT count(*) FROM (SELECT fts_main_data.match_bm25(__hf_index_id, 'world') AS s FROM data)"
            " WHERE s IS NOT NULL"
        ).fetchone()
    assert hits == (100,)


def test_create_table_reads_parquet_files_with_different_schemas(tmp_path_factory: pytest.TempPathFactory) -> None:
    directory = tmp_path_factory.mktemp("data")
    # a column missing from a file, a column all null in a file and a string in another one
    pq.write_table(
        pa.table({"text": ["a"], "source": ["x"], "target": pa.array([None], pa.null())}), directory / "0.parquet"
    )  # type: ignore
    pq.write_table(pa.table({"text": ["b"], "target": ["t"]}), directory / "1.parquet")  # type: ignore
    files = [directory / "0.parquet", directory / "1.parquet"]
    source = "[" + ",".join(varchar_sql(str(path)) for path in files) + "]"
    with duckdb_connect(database=str(directory / "index.duckdb"), allowed_paths=files) as con:
        con.sql(
            CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES.format(columns='"text","source","target"', source=source)
        )
        rows = con.sql('SELECT "text", "source", "target" FROM data ORDER BY "text"').fetchall()
    assert rows == [("a", "x", None), ("b", None, "t")]


def test_compute_length_column_when_a_file_misses_the_column(tmp_path_factory: pytest.TempPathFactory) -> None:
    directory = tmp_path_factory.mktemp("data")
    pq.write_table(pa.table({"text": ["ab"], "source": ["xyz"]}), directory / "0.parquet")  # type: ignore
    pq.write_table(pa.table({"text": ["c"]}), directory / "1.parquet")  # type: ignore

    length_column = compute_length_column(
        [directory / "0.parquet", directory / "1.parquet"], "source", target_df=None, dtype="string"
    )
    assert length_column.to_dict(as_series=False) == {"source.length": [3, None]}
