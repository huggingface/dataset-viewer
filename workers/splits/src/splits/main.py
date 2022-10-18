# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcache.simple_cache import connect_to_cache
from libcommon.logger import init_logger
from libqueue.queue import connect_to_queue

from splits.config import WorkerConfig
from splits.worker import SplitsWorker

if __name__ == "__main__":
    worker_config = WorkerConfig()
    init_logger(worker_config.common.log_level)
    connect_to_cache(database=worker_config.cache.mongo_database, host=worker_config.cache.mongo_url)
    connect_to_queue(database=worker_config.queue.mongo_database, host=worker_config.cache.mongo_url)

    SplitsWorker(worker_config).loop()
