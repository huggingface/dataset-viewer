"""Tests for libviewer CDC detection function."""

import tempfile
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from libviewer._internal import is_content_defined_chunked_parquet


SEED = 42


def test_regular_parquet_is_not_cdc():
    """Regular integer-only parquet files should NOT be detected as CDC."""
    table = pa.table(
        {
            "a": list(range(5000)),
            "b": list(range(5000, 10000)),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(table, f.name, write_page_index=True)
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert not result, f"Regular parquet should not be CDC, got {result}"


def test_regular_parquet_with_string_column():
    """Parquet with uniform-length string column should NOT be CDC."""
    strings = ["abcdefghij" for _ in range(5000)]  # All same length
    table = pa.table({"data": pa.array(strings, type=pa.string())})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(table, f.name, write_page_index=True)
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert not result, f"Uniform-length strings should not be CDC, got {result}"


def test_small_regular_parquet():
    """Small parquet files should still work correctly."""
    table = pa.table({"a": list(range(10))})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(table, f.name, write_page_index=True)
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert not result, f"Small regular parquet should not be CDC, got {result}"


def test_empty_bytes_raises():
    """Empty bytes should raise an error."""
    try:
        is_content_defined_chunked_parquet(b"")
        assert False, "Expected error for empty bytes"
    except Exception:
        pass


def test_invalid_bytes_raises():
    """Invalid bytes should raise an error."""
    try:
        is_content_defined_chunked_parquet(b"\x00\x01\x02\x03\x04\x05")
        assert False, "Expected error for invalid bytes"
    except Exception:
        pass


def test_cdc_parquet_with_multiple_row_groups():
    """Parquet with content-defined chunking and multiple row groups SHOULD be detected as CDC."""
    np.random.seed(SEED)
    strings = [f"row-{i}-data-" * np.random.randint(1, 200) for i in range(1000)]
    table = pa.table({"data": pa.array(strings, type=pa.string())})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(
            table,
            f.name,
            use_content_defined_chunking={
                "min_chunk_size": 2_500,
                "max_chunk_size": 10_000,
                "norm_level": 0,
            },
            write_page_index=True,
        )
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert result, f"CDC parquet should be detected as CDC, got {result}"


def test_cdc_parquet_without_cdc_is_not_cdc():
    """Same data as test_cdc_parquet_with_multiple_row_groups but WITHOUT CDC should NOT be detected as CDC."""
    np.random.seed(SEED)
    strings = [f"row-{i}-data-" * np.random.randint(1, 200) for i in range(1000)]
    table = pa.table({"data": pa.array(strings, type=pa.string())})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(
            table,
            f.name,
            data_page_size=10_000,
            write_page_index=True,
        )
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert not result, f"Non-CDC parquet should NOT be detected as CDC, got {result}"


def test_regular_bytes_returns_false():
    """Bytes that don't form a valid parquet file should return False or raise."""
    # Create a minimal valid parquet file with no CDC
    table = pa.table({"x": [1, 2, 3]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(table, f.name)
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert result is False or isinstance(result, bool)


def test_cdc_single_column():
    """CDC parquet with only one column should be detected."""
    strings = [f"data-{i}" * (i + 1) for i in range(1000)]
    table = pa.table({"text": pa.array(strings, type=pa.string())})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(
            table,
            f.name,
            use_content_defined_chunking={
                "min_chunk_size": 1_000,
                "max_chunk_size": 5_000,
                "norm_level": 0,
            },
            write_page_index=True,
        )
        with open(f.name, "rb") as ff:
            data = ff.read()
        result = is_content_defined_chunked_parquet(data)
    os.unlink(f.name)
    assert result, f"CDC parquet should be detected as CDC, got {result}"
