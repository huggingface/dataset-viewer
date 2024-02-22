from typing import Any

import duckdb

LOAD_FTS_SAFE_COMMAND = "LOAD 'fts'; SET enable_external_access=false; SET lock_configuration=true;"


def duckdb_connect(**kwargs: Any) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(read_only=True, **kwargs)
    con.sql(LOAD_FTS_SAFE_COMMAND)
    return con
