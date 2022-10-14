# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from libcache.simple_cache import _clean_database as _clean_cache_database
from libcache.simple_cache import connect_to_cache
from libqueue.queue import _clean_queue_database, connect_to_queue

from worker.config import (
    HF_ENDPOINT,
    HF_TOKEN,
    MAX_JOBS_PER_DATASET,
    MAX_LOAD_PCT,
    MAX_MEMORY_PCT,
    WORKER_SLEEP_SECONDS,
)
from worker.workers.splits import SplitsWorker

from ..utils import MONGO_CACHE_DATABASE, MONGO_QUEUE_DATABASE, MONGO_URL


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


def test_first_rows_worker(hub_public_csv: str) -> None:
    worker = SplitsWorker(
        hf_endpoint=HF_ENDPOINT,
        hf_token=HF_TOKEN,
        max_jobs_per_dataset=MAX_JOBS_PER_DATASET,
        max_load_pct=MAX_LOAD_PCT,
        max_memory_pct=MAX_MEMORY_PCT,
        sleep_seconds=WORKER_SLEEP_SECONDS,
    )
    worker.queues.splits.add_job(dataset=hub_public_csv)
    result = worker.process_next_job()
    assert result is True
