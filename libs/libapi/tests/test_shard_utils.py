# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Authors.

import pytest

from libapi.shard_utils import (
    get_original_shard_for_row,
    get_parquet_shard_for_row,
    get_shard_info,
)


def test_get_original_shard_for_row_standard() -> None:
    """Row 150 with lengths [100, 100, 100, 100] -> shard 1, rows 100-199"""
    result = get_original_shard_for_row(150, [100, 100, 100, 100])
    assert result["original_shard_index"] == 1
    assert result["original_shard_start_row"] == 100
    assert result["original_shard_end_row"] == 199


def test_get_original_shard_for_row_first_shard() -> None:
    """Row 0 -> shard 0"""
    result = get_original_shard_for_row(0, [100, 100, 100, 100])
    assert result["original_shard_index"] == 0
    assert result["original_shard_start_row"] == 0
    assert result["original_shard_end_row"] == 99


def test_get_original_shard_for_row_last_row() -> None:
    """Row 399 (last) -> shard 3"""
    result = get_original_shard_for_row(399, [100, 100, 100, 100])
    assert result["original_shard_index"] == 3


def test_get_original_shard_for_row_boundary() -> None:
    """Row 100 (exactly at boundary) -> shard 1, not shard 0"""
    result = get_original_shard_for_row(100, [100, 100, 100, 100])
    assert result["original_shard_index"] == 1
    assert result["original_shard_start_row"] == 100


def test_get_parquet_shard_for_row_single() -> None:
    """Single shard dataset -> always index 0"""
    parquet_files = [{"filename": "train.parquet", "split": "train"}]
    result = get_parquet_shard_for_row(50, None, parquet_files, "train")
    assert result["parquet_shard_index"] == 0


def test_get_parquet_shard_for_row_multi() -> None:
    """Multi-shard with lengths [200, 200] -> correct index"""
    parquet_files = [
        {"filename": "train-00000-of-00002.parquet", "split": "train"},
        {"filename": "train-00001-of-00002.parquet", "split": "train"},
    ]
    result = get_parquet_shard_for_row(150, [200, 200], parquet_files, "train")
    assert result["parquet_shard_index"] == 0
    result = get_parquet_shard_for_row(250, [200, 200], parquet_files, "train")
    assert result["parquet_shard_index"] == 1


def test_missing_original_shard_lengths() -> None:
    """Legacy dataset - key MISSING (not null) -> graceful handling"""
    split_info = {"num_examples": 400, "shard_lengths": [200, 200]}
    # Note: "original_shard_lengths" key is MISSING, not None
    parquet_files = [
        {"filename": "train-00000-of-00002.parquet", "split": "train"},
        {"filename": "train-00001-of-00002.parquet", "split": "train"},
    ]
    result = get_shard_info(150, split_info, parquet_files, "train")
    assert result["original_shard_index"] is None
    assert result["original_shard_info"] is not None  # Contains explanation


def test_corrupted_metadata_sum_mismatch() -> None:
    """sum(original_shard_lengths) != num_examples -> ValueError"""
    split_info = {
        "num_examples": 400,
        "shard_lengths": [200, 200],
        "original_shard_lengths": [100, 100],  # Sum = 200 != 400
    }
    parquet_files = [{"filename": "train.parquet", "split": "train"}]
    with pytest.raises(ValueError, match="Corrupted metadata"):
        get_shard_info(150, split_info, parquet_files, "train")


def test_row_out_of_bounds() -> None:
    """row >= num_examples -> IndexError"""
    split_info = {"num_examples": 400, "shard_lengths": [200, 200]}
    parquet_files = [{"filename": "train.parquet", "split": "train"}]
    with pytest.raises(IndexError):
        get_shard_info(500, split_info, parquet_files, "train")


def test_no_parquet_files_for_split() -> None:
    """No parquet files for split -> ValueError (not silent fallback)"""
    parquet_files: list[dict[str, str]] = []  # Empty!
    with pytest.raises(ValueError, match="No parquet files found"):
        get_parquet_shard_for_row(0, [100], parquet_files, "train")


def test_metadata_inconsistency_more_shards_than_files() -> None:
    """More shards in shard_lengths than parquet files -> ValueError"""
    parquet_files = [{"filename": "train-00000.parquet", "split": "train"}]  # Only 1 file
    shard_lengths = [100, 100, 100]  # But 3 shards
    # Row 150 would be in shard index 1, but only 1 file exists
    with pytest.raises(ValueError, match="Metadata inconsistency"):
        get_parquet_shard_for_row(150, shard_lengths, parquet_files, "train")
