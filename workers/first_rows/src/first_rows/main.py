# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcache.asset import show_assets_dir
from libcache.simple_cache import connect_to_cache
from libqueue.queue import connect_to_queue
from libutils.logger import init_logger

from first_rows.config import (
    ASSETS_BASE_URL,
    ASSETS_DIRECTORY,
    HF_ENDPOINT,
    HF_TOKEN,
    LOG_LEVEL,
    MAX_JOBS_PER_DATASET,
    MAX_LOAD_PCT,
    MAX_MEMORY_PCT,
    MAX_SIZE_FALLBACK,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
    ROWS_MAX_BYTES,
    ROWS_MAX_NUMBER,
    ROWS_MIN_NUMBER,
    WORKER_SLEEP_SECONDS,
)
from first_rows.worker import FirstRowsWorker

if __name__ == "__main__":
    init_logger(LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    show_assets_dir(ASSETS_DIRECTORY)
    FirstRowsWorker(
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
    ).loop()
