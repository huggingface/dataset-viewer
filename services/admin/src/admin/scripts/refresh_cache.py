# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from huggingface_hub.hf_api import HfApi  # type: ignore
from libcommon.logger import init_logger
from libqueue.queue import Queue, connect_to_queue

from admin.config import AppConfig
from admin.utils import JobType


def get_hf_dataset_names(hf_endpoint: str):
    return [str(dataset.id) for dataset in HfApi(hf_endpoint).list_datasets(full=False)]


def refresh_datasets_cache(dataset_names: List[str]) -> None:
    logger = logging.getLogger("refresh_cache")
    splits_queue = Queue(type=JobType.SPLITS.value)
    for dataset_name in dataset_names:
        # don't mark the cache entries as stale, because it's manually triggered
        splits_queue.add_job(dataset=dataset_name)
        logger.info(f"added a job to refresh '{dataset_name}'")


if __name__ == "__main__":
    app_config = AppConfig()
    init_logger(app_config.common.log_level, "refresh_cache")
    logger = logging.getLogger("refresh_cache")
    connect_to_queue(database=app_config.queue.mongo_database, host=app_config.cache.mongo_url)
    refresh_datasets_cache(get_hf_dataset_names(hf_endpoint=app_config.common.hf_endpoint))
    logger.info("all the datasets of the Hub have been added to the queue to refresh the cache")
