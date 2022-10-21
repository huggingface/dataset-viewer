# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcache.simple_cache import get_datasets_with_some_error

from admin.config import AppConfig
from admin.scripts.refresh_cache import refresh_datasets_cache

if __name__ == "__main__":
    app_config = AppConfig()
    refresh_datasets_cache(get_datasets_with_some_error())
    logging.info("all the datasets with some error in the cache have been added to the queue to be refreshed")
