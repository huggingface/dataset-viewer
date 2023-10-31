from typing import Any

import duckdb


def duckdb_connect(**kwargs: Any) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(read_only=True, **kwargs)
    con.sql("SET enable_external_access=false;")
    con.sql("SET lock_configuration=true;")
    return con
