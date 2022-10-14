# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from huggingface_hub.hf_api import HfApi  # type: ignore
from libqueue.queue import Queue, connect_to_queue
from libutils.logger import init_logger

from ..config import HF_ENDPOINT, LOG_LEVEL, MONGO_QUEUE_DATABASE, MONGO_URL
from ..utils import JobType


def get_hf_dataset_names():
    return [str(dataset.id) for dataset in HfApi(HF_ENDPOINT).list_datasets(full=False)]


def refresh_datasets_cache(dataset_names: List[str]) -> None:
    logger = logging.getLogger("refresh_cache")
    splits_queue = Queue(type=JobType.SPLITS.value)
    connect_to_queue(MONGO_QUEUE_DATABASE, MONGO_URL)
    for dataset_name in dataset_names:
        # don't mark the cache entries as stale, because it's manually triggered
        splits_queue.add_job(dataset=dataset_name)
        logger.info(f"added a job to refresh '{dataset_name}'")


if __name__ == "__main__":
    init_logger(LOG_LEVEL, "refresh_cache")
    logger = logging.getLogger("refresh_cache")
    refresh_datasets_cache(get_hf_dataset_names())
    logger.info("all the datasets of the Hub have been added to the queue to refresh the cache")
