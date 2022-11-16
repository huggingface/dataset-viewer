# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from huggingface_hub.hf_api import HfApi

from admin.config import AppConfig
from admin.scripts.refresh_cache import refresh_datasets_cache


def get_hf_canonical_dataset_names(hf_endpoint: str):
    return [str(dataset.id) for dataset in HfApi(hf_endpoint).list_datasets(full=False) if dataset.id.find("/") == -1]


if __name__ == "__main__":
    app_config = AppConfig()
    refresh_datasets_cache(get_hf_canonical_dataset_names(hf_endpoint=app_config.common.hf_endpoint))
    logging.info("all the canonical datasets of the Hub have been added to the queue to refresh the cache")
