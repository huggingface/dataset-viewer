from libcommon.duckdb_utils import duckdb_index_is_partial


def test_duckdb_index_is_partial() -> None:
    assert duckdb_index_is_partial(
        "https://hf.co/datasets/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-train/index.duckdb"
    )
    assert duckdb_index_is_partial(
        "https://hf.co/datasets/bigcode/the-stack/resolve/refs%2Fconvert%2Fparquet/default/train/partial-index.duckdb"
    )
    assert not duckdb_index_is_partial(
        "https://hf.co/datasets/squad/resolve/refs%2Fconvert%2Fparquet/plain_text/train/index.duckdb"
    )
