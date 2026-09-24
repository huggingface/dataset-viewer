# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Authors.

from typing import Any, Optional, TypedDict


class OriginalShardInfo(TypedDict):
    original_shard_index: int
    original_shard_start_row: int
    original_shard_end_row: int


class ParquetShardInfo(TypedDict):
    parquet_shard_index: int
    parquet_shard_file: str


class ShardInfoResponse(TypedDict):
    row_index: int
    original_shard_index: Optional[int]
    original_shard_start_row: Optional[int]
    original_shard_end_row: Optional[int]
    original_shard_info: Optional[str]  # Explanation when missing
    parquet_shard_index: int
    parquet_shard_file: str


def get_original_shard_for_row(row_index: int, original_shard_lengths: list[int]) -> OriginalShardInfo:
    """Cumulative sum to find which original shard contains row."""
    cumulative = 0
    for shard_idx, length in enumerate(original_shard_lengths):
        if cumulative + length > row_index:
            return {
                "original_shard_index": shard_idx,
                "original_shard_start_row": cumulative,
                "original_shard_end_row": cumulative + length - 1,
            }
        cumulative += length
    raise IndexError(f"row_index {row_index} out of bounds")


def get_parquet_shard_for_row(
    row_index: int,
    shard_lengths: Optional[list[int]],
    parquet_files: list[dict[str, Any]],
    split: str,
) -> ParquetShardInfo:
    """Find which parquet file contains the row."""
    # Filter and sort parquet files for this split
    split_files = sorted(
        [f for f in parquet_files if f.get("split") == split],
        key=lambda f: f["filename"],
    )

    if not split_files:
        # Following codebase pattern (duckdb.py:97-98): raise error, don't fabricate
        raise ValueError(f"No parquet files found for split '{split}'")

    if not shard_lengths or len(shard_lengths) <= 1:
        # Single shard
        return {
            "parquet_shard_index": 0,
            "parquet_shard_file": split_files[0]["filename"],
        }

    cumulative = 0
    for shard_idx, length in enumerate(shard_lengths):
        if cumulative + length > row_index:
            if shard_idx >= len(split_files):
                # Metadata inconsistency: more shards than parquet files
                raise ValueError(
                    f"Metadata inconsistency: shard_lengths has {len(shard_lengths)} shards "
                    f"but only {len(split_files)} parquet files for split '{split}'"
                )
            return {
                "parquet_shard_index": shard_idx,
                "parquet_shard_file": split_files[shard_idx]["filename"],
            }
        cumulative += length
    raise IndexError(f"row_index {row_index} out of bounds")


def get_shard_info(
    row_index: int,
    split_info: dict[str, Any],
    parquet_files: list[dict[str, Any]],
    split: str,
) -> ShardInfoResponse:
    """
    Main entry point: compute shard info for a row.

    CRITICAL: Check for key EXISTENCE, not nullity!
    The field is MISSING from JSON when None, not null.
    """
    num_examples = split_info.get("num_examples", 0)

    # Validate bounds
    if row_index < 0 or row_index >= num_examples:
        raise IndexError(f"row_index {row_index} out of bounds (0-{num_examples - 1})")

    # Always compute parquet shard (from shard_lengths)
    shard_lengths = split_info.get("shard_lengths")

    # Validate shard_lengths integrity if present (catches corrupted parquet metadata)
    if shard_lengths and sum(shard_lengths) != num_examples:
        raise ValueError(f"Corrupted metadata: sum(shard_lengths)={sum(shard_lengths)} != num_examples={num_examples}")

    parquet_info = get_parquet_shard_for_row(row_index, shard_lengths, parquet_files, split)

    # Check for original_shard_lengths - KEY EXISTENCE, not nullity!
    if "original_shard_lengths" not in split_info:
        return {
            "row_index": row_index,
            "original_shard_index": None,
            "original_shard_start_row": None,
            "original_shard_end_row": None,
            "original_shard_info": "not available - dataset predates shard tracking or has single input shard",
            **parquet_info,
        }

    original_shard_lengths = split_info["original_shard_lengths"]

    # Sanity check: validate metadata integrity
    total_original = sum(original_shard_lengths)
    if total_original != num_examples:
        raise ValueError(
            f"Corrupted metadata: sum(original_shard_lengths)={total_original} != num_examples={num_examples}"
        )

    original_info = get_original_shard_for_row(row_index, original_shard_lengths)

    return {
        "row_index": row_index,
        "original_shard_index": original_info["original_shard_index"],
        "original_shard_start_row": original_info["original_shard_start_row"],
        "original_shard_end_row": original_info["original_shard_end_row"],
        "original_shard_info": None,
        **parquet_info,
    }
