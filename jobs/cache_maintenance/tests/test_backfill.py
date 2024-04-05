# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from unittest.mock import patch

import pytest
from libcommon.operations import OperationsStatistics

from cache_maintenance.backfill import backfill_datasets, try_backfill_dataset

from .constants import CI_APP_TOKEN, CI_HUB_ENDPOINT


@pytest.mark.parametrize(
    "num_backfilled_datasets,num_untouched_datasets,num_deleted_datasets",
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ],
)
def test_try_backfill_dataset(
    num_backfilled_datasets: int, num_untouched_datasets: int, num_deleted_datasets: int
) -> None:
    with patch(
        "cache_maintenance.backfill.backfill_dataset",
        return_value=OperationsStatistics(
            num_backfilled_datasets=num_backfilled_datasets,
            num_untouched_datasets=num_untouched_datasets,
            num_deleted_datasets=num_deleted_datasets,
        ),
    ):
        statistics = try_backfill_dataset(
            dataset="dataset",
            hf_endpoint=CI_HUB_ENDPOINT,
            blocked_datasets=[],
            hf_token=CI_APP_TOKEN,
            storage_clients=[],
        )
    assert statistics.num_total_datasets == 0
    assert statistics.num_analyzed_datasets == 1
    assert statistics.num_error_datasets == 0
    assert statistics.operations.num_backfilled_datasets == num_backfilled_datasets
    assert statistics.operations.num_untouched_datasets == num_untouched_datasets
    assert statistics.operations.num_deleted_datasets == num_deleted_datasets


def test_try_backfill_dataset_exception() -> None:
    with patch("cache_maintenance.backfill.backfill_dataset", side_effect=Exception("error")):
        statistics = try_backfill_dataset(
            dataset="dataset",
            hf_endpoint=CI_HUB_ENDPOINT,
            blocked_datasets=[],
            hf_token=CI_APP_TOKEN,
            storage_clients=[],
        )
    assert statistics.num_analyzed_datasets == 1
    assert statistics.num_error_datasets == 1


def test_backfill_datasets() -> None:
    num_datasets = 200
    with patch(
        "cache_maintenance.backfill.backfill_dataset",
        return_value=OperationsStatistics(
            num_backfilled_datasets=1,
            num_untouched_datasets=0,
            num_deleted_datasets=0,
        ),
    ):
        statistics = backfill_datasets(
            dataset_names={f"a{i}" for i in range(0, num_datasets)},
            hf_endpoint=CI_HUB_ENDPOINT,
            blocked_datasets=[],
            hf_token=CI_APP_TOKEN,
            storage_clients=[],
        )
    assert statistics.num_total_datasets == num_datasets
    assert statistics.num_analyzed_datasets == num_datasets
    assert statistics.num_error_datasets == 0
    assert statistics.operations.num_backfilled_datasets == num_datasets
    assert statistics.operations.num_untouched_datasets == 0
    assert statistics.operations.num_deleted_datasets == 0
