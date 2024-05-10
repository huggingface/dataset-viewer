# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
import pytest

from libcommon.orchestrator import SmartUpdateImpossibleBecauseCacheIsEmpty
from libcommon.resources import CacheMongoResource, QueueMongoResource

from .utils import (
    DATASET_NAME,
    PROCESSING_GRAPH_TWO_STEPS,
    REVISION_NAME,
    STEP_DA,
    assert_smart_dataset_update_plan,
    get_smart_dataset_update_plan,
    put_cache,
    put_diff,
)

EMPTY_DIFF = ""


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


def test_initial_state() -> None:
    with pytest.raises(SmartUpdateImpossibleBecauseCacheIsEmpty):
        get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME)
    with put_diff(EMPTY_DIFF):
        plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
        assert_smart_dataset_update_plan(plan, tasks=[])
