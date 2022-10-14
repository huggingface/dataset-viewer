# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest
from libcache.simple_cache import _clean_database as _clean_cache_database
from libcache.simple_cache import connect_to_cache
from libqueue.queue import _clean_queue_database, connect_to_queue

from worker.config import (
    ASSETS_BASE_URL,
    HF_ENDPOINT,
    HF_TOKEN,
    MAX_JOBS_PER_DATASET,
    MAX_LOAD_PCT,
    MAX_MEMORY_PCT,
    MAX_SIZE_FALLBACK,
    ROWS_MAX_BYTES,
    ROWS_MAX_NUMBER,
    ROWS_MIN_NUMBER,
    WORKER_SLEEP_SECONDS,
)
from worker.workers.first_rows import FirstRowsWorker

from ..utils import (
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


def test_first_rows_worker(hub_public_csv: str) -> None:
    dataset, config, split = get_default_config_split(hub_public_csv)
    worker = FirstRowsWorker(
        assets_base_url=ASSETS_BASE_URL,
        hf_endpoint=HF_ENDPOINT,
        hf_token=HF_TOKEN,
        max_size_fallback=MAX_SIZE_FALLBACK,
        rows_max_bytes=ROWS_MAX_BYTES,
        rows_max_number=ROWS_MAX_NUMBER,
        rows_min_number=ROWS_MIN_NUMBER,
        max_jobs_per_dataset=MAX_JOBS_PER_DATASET,
        max_load_pct=MAX_LOAD_PCT,
        max_memory_pct=MAX_MEMORY_PCT,
        sleep_seconds=WORKER_SLEEP_SECONDS,
    )
    worker.queues.first_rows.add_job(dataset=dataset, config=config, split=split)
    result = worker.process_next_job()
    assert result is True
