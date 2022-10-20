# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcache.simple_cache import connect_to_cache, get_datasets_with_some_error
from libcommon.logger import init_logger
from libqueue.queue import connect_to_queue

from admin.config import AppConfig
from admin.scripts.refresh_cache import refresh_datasets_cache

if __name__ == "__main__":
    app_config = AppConfig()
    init_logger(app_config.common.log_level, "refresh_cache_canonical")
    logger = logging.getLogger("refresh_cache_canonical")
    connect_to_cache(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    connect_to_queue(database=app_config.queue.mongo_database, host=app_config.cache.mongo_url)
    refresh_datasets_cache(get_datasets_with_some_error())
    logger.info("all the datasets with some error in the cache have been added to the queue to be refreshed")
