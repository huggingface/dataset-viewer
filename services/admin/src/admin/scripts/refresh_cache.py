# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from huggingface_hub.hf_api import HfApi
from libqueue.queue import Queue

from admin.config import AppConfig
from admin.utils import JobType


def get_hf_dataset_names(hf_endpoint: str):
    return [str(dataset.id) for dataset in HfApi(hf_endpoint).list_datasets(full=False)]


def refresh_datasets_cache(dataset_names: List[str]) -> None:
    splits_queue = Queue(type=JobType.SPLITS.value)
    for dataset_name in dataset_names:
        # don't mark the cache entries as stale, because it's manually triggered
        splits_queue.add_job(dataset=dataset_name)
        logging.info(f"added a job to refresh '{dataset_name}'")


if __name__ == "__main__":
    app_config = AppConfig()
    refresh_datasets_cache(get_hf_dataset_names(hf_endpoint=app_config.common.hf_endpoint))
    logging.info("all the datasets of the Hub have been added to the queue to refresh the cache")
