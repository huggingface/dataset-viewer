# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcache.simple_cache import connect_to_cache, get_datasets_with_some_error
from libutils.logger import init_logger

from ..config import LOG_LEVEL, MONGO_CACHE_DATABASE, MONGO_URL
from .refresh_cache import refresh_datasets_cache

if __name__ == "__main__":
    init_logger(LOG_LEVEL, "refresh_cache_canonical")
    logger = logging.getLogger("refresh_cache_canonical")
    connect_to_cache(MONGO_CACHE_DATABASE, MONGO_URL)
    refresh_datasets_cache(get_datasets_with_some_error())
    logger.info("all the datasets with some error in the cache have been added to the queue to be refreshed")
