from typing import Any, Optional

import duckdb

ATTACH_READ_ONLY_DATABASE = "ATTACH '{database}' as db (READ_ONLY); USE db;"
LOAD_FTS_SAFE_COMMAND = "INSTALL 'fts'; LOAD 'fts'; SET enable_external_access=false; SET lock_configuration=true;"
SET_EXTENSIONS_DIRECTORY_COMMAND = "SET extension_directory='{directory}';"


def duckdb_connect(
    database: Optional[str] = None, extensions_directory: Optional[str] = None, **kwargs: Any
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:", **kwargs)
    if database is not None:
        con.execute(ATTACH_READ_ONLY_DATABASE.format(database=database))
    if extensions_directory is not None:
        con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))
    con.sql(LOAD_FTS_SAFE_COMMAND)
    return con
