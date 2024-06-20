# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.queue.dataset_blockages import block_dataset, get_blocked_datasets
from libcommon.resources import QueueMongoResource


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.mark.parametrize(
    "datasets,expected_datasets",
    [
        ([], set()),
        (["dataset"], {"dataset"}),
        (["dataset", "dataset"], {"dataset"}),
        (["dataset1", "dataset2"], {"dataset1", "dataset2"}),
    ],
)
def test_dataset_blockage(datasets: list[str], expected_datasets: set[str]) -> None:
    for dataset in datasets:
        block_dataset(dataset=dataset)

    assert get_blocked_datasets() == expected_datasets
