# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcache.simple_cache import connect_to_cache
from libqueue.queue import connect_to_queue
from libutils.logger import init_logger

from splits.config import (
    HF_ENDPOINT,
    HF_TOKEN,
    LOG_LEVEL,
    MAX_JOBS_PER_DATASET,
    MAX_LOAD_PCT,
    MAX_MEMORY_PCT,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
    WORKER_SLEEP_SECONDS,
)
from splits.worker import SplitsWorker

if __name__ == "__main__":
    init_logger(LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    SplitsWorker(
        hf_endpoint=HF_ENDPOINT,
        hf_token=HF_TOKEN,
        max_jobs_per_dataset=MAX_JOBS_PER_DATASET,
        max_load_pct=MAX_LOAD_PCT,
        max_memory_pct=MAX_MEMORY_PCT,
        sleep_seconds=WORKER_SLEEP_SECONDS,
    ).loop()
