# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.queue.dataset_blockages import block_dataset, get_blocked_datasets, is_blocked
from libcommon.resources import QueueMongoResource


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.mark.parametrize(
    "datasets,expected_datasets",
    [
        ([], []),
        (["dataset"], ["dataset"]),
        (["dataset", "dataset"], ["dataset"]),
        (["dataset1", "dataset2"], ["dataset1", "dataset2"]),
    ],
)
def test_dataset_blockage(datasets: list[str], expected_datasets: set[str]) -> None:
    for dataset in datasets:
        block_dataset(dataset=dataset)

    assert sorted(get_blocked_datasets()) == sorted(expected_datasets)
    for dataset in expected_datasets:
        assert is_blocked(dataset=dataset)
    assert not is_blocked(dataset="not_blocked_dataset")
