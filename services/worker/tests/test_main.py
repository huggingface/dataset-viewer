# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from libcache.simple_cache import _clean_database as _clean_cache_database
from libcache.simple_cache import connect_to_cache
from libqueue.queue import _clean_queue_database, add_job, connect_to_queue

from worker.main import process_next_first_rows_job, process_next_splits_job
from worker.utils import JobType

from .utils import (
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
    get_default_config_split,
)


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_cache_database()
    _clean_queue_database()


def test_process_next_splits_job(hub_public_csv: str) -> None:
    add_job(JobType.SPLITS.value, dataset=hub_public_csv)
    result = process_next_splits_job()
    assert result is True


def test_process_next_first_rows_job(hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    add_job(JobType.FIRST_ROWS.value, dataset=dataset, config=config, split=split)
    result = process_next_first_rows_job()
    assert result is True
