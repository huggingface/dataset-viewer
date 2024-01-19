from typing import Any

import duckdb


def duckdb_connect(**kwargs: Any) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(read_only=True, **kwargs)
    # TODO: Temporary commented because it raises
    # duckdb.duckdb.Error: Extension Autoloading Error:
    # An error occurred while trying to automatically install the required extension
    # con.sql("SET enable_external_access=false;")
    con.sql("SET lock_configuration=true;")
    return con
