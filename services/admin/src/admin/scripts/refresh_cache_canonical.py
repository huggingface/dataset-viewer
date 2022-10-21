# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from huggingface_hub.hf_api import HfApi  # type: ignore
from libcommon.logger import init_logger
from libqueue.queue import connect_to_queue

from admin.config import AppConfig
from admin.scripts.refresh_cache import refresh_datasets_cache


def get_hf_canonical_dataset_names(hf_endpoint: str):
    return [str(dataset.id) for dataset in HfApi(hf_endpoint).list_datasets(full=False) if dataset.id.find("/") == -1]


if __name__ == "__main__":
    app_config = AppConfig()
    init_logger(app_config.common.log_level, "refresh_cache_canonical")
    logger = logging.getLogger("refresh_cache_canonical")
    connect_to_queue(database=app_config.queue.mongo_database, host=app_config.cache.mongo_url)
    refresh_datasets_cache(get_hf_canonical_dataset_names(hf_endpoint=app_config.common.hf_endpoint))
    logger.info("all the canonical datasets of the Hub have been added to the queue to refresh the cache")
