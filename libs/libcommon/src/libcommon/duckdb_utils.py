from libcommon.parquet_utils import PARTIAL_PREFIX, parquet_export_is_partial


def duckdb_index_is_partial(duckdb_index_url: str) -> bool:
    """
    Check if the DuckDB index is on the full dataset or if it's partial.
    It could be partial for two reasons:

    1. if the Parquet export that was used to build it is partial
    2. if it's a partial index of the Parquet export

    Args:
        duckdb_index_url (str): The URL of the DuckDB index file.

    Returns:
        partial (bool): True is the DuckDB index is partial,
            or False if it's an index of the full dataset.
    """
    _, duckdb_index_file_name = duckdb_index_url.rsplit("/", 1)
    return parquet_export_is_partial(duckdb_index_url) or duckdb_index_file_name.startswith(PARTIAL_PREFIX)
