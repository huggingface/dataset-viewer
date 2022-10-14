# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from huggingface_hub.hf_api import HfApi  # type: ignore
from libutils.logger import init_logger

from ..config import HF_ENDPOINT, LOG_LEVEL
from .refresh_cache import refresh_datasets_cache


def get_hf_canonical_dataset_names():
    return [str(dataset.id) for dataset in HfApi(HF_ENDPOINT).list_datasets(full=False) if dataset.id.find("/") == -1]


if __name__ == "__main__":
    init_logger(LOG_LEVEL, "refresh_cache_canonical")
    logger = logging.getLogger("refresh_cache_canonical")
    refresh_datasets_cache(get_hf_canonical_dataset_names())
    logger.info("all the canonical datasets of the Hub have been added to the queue to refresh the cache")
