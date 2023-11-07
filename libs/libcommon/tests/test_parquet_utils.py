from libcommon.parquet_utils import parquet_export_is_partial


def test_parquet_export_is_partial() -> None:
    assert parquet_export_is_partial(
        "https://hf.co/datasets/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-train/0000.parquet"
    )
    assert not parquet_export_is_partial(
        "https://hf.co/datasets/bigcode/the-stack/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    assert not parquet_export_is_partial(
        "https://hf.co/datasets/squad/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet"
    )
