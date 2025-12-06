# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Authors.

from .utils import get_default_config_split, poll_until_ready_and_assert


def test_shard_endpoint(normal_user_public_dataset: str) -> None:
    """Test /shard endpoint returns correct shard info."""
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()

    # First ensure dataset is processed (wait for config-parquet-and-info)
    poll_until_ready_and_assert(
        relative_url=f"/info?dataset={dataset}&config={config}",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
    )

    # Now test shard endpoint
    shard_response = poll_until_ready_and_assert(
        relative_url=f"/shard?dataset={dataset}&config={config}&split={split}&row=0",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
    )

    content = shard_response.json()
    assert "row_index" in content
    assert content["row_index"] == 0
    assert "parquet_shard_index" in content
    assert "parquet_shard_file" in content
    # original_shard_index may be None for legacy/single-shard datasets


def test_shard_endpoint_row_out_of_bounds(normal_user_public_dataset: str) -> None:
    """Test /shard endpoint returns 400 for invalid row."""
    dataset = normal_user_public_dataset
    config, split = get_default_config_split()

    # First ensure dataset is processed
    poll_until_ready_and_assert(
        relative_url=f"/info?dataset={dataset}&config={config}",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
    )

    # Test with row out of bounds
    poll_until_ready_and_assert(
        relative_url=f"/shard?dataset={dataset}&config={config}&split={split}&row=999999",
        expected_status_code=400,
        expected_error_code="RowOutOfBounds",
        dataset=dataset,
        should_retry_x_error_codes=["ResponseNotFound"],
    )
