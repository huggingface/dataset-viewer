from typing import Any, Optional

import duckdb

LOAD_FTS_SAFE_COMMAND = "INSTALL 'fts'; LOAD 'fts'; SET enable_external_access=false; SET lock_configuration=true;"
SET_EXTENSIONS_DIRECTORY_COMMAND = "SET extension_directory='{directory}';"


def duckdb_connect(extensions_directory: Optional[str] = None, **kwargs: Any) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(read_only=True, **kwargs)
    if extensions_directory is not None:
        con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))
    con.sql(LOAD_FTS_SAFE_COMMAND)
    return con
