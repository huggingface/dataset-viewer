from libcommon.duckdb_utils import duckdb_index_is_partial


def test_duckdb_index_is_partial() -> None:
    assert duckdb_index_is_partial(
        "https://hf.co/datasets/canonical/resolve/refs%2Fconvert%2Fduckdb/en/partial-train/index.duckdb"
    )
    assert duckdb_index_is_partial(
        "https://hf.co/datasets/organization/not-canonical/resolve/refs%2Fconvert%2Fduckdb/default/train/partial-index.duckdb"
    )
    assert not duckdb_index_is_partial(
        "https://hf.co/datasets/rajpurkar/squad/resolve/refs%2Fconvert%2Fduckdb/plain_text/train/index.duckdb"
    )
