# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from libcache.simple_cache import (
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
)
from libqueue.queue import Queue

from admin.utils import JobType

splits_queue = Queue(type=JobType.SPLITS.value)
first_rows_queue = Queue(type=JobType.FIRST_ROWS.value)


def is_supported(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Check if the dataset exists on the Hub and is supported by the datasets-server.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        [`bool`]: True if the dataset is supported by the datasets-server.
    """
    try:
        # note that token is required to access gated dataset info
        info = HfApi(endpoint=hf_endpoint).dataset_info(dataset, token=hf_token)
    except RepositoryNotFoundError:
        return False
    return info.private is False


def update_splits(dataset: str, force: bool = False) -> None:
    logging.debug(f"refresh /splits for {dataset}")
    mark_splits_responses_as_stale(dataset_name=dataset)
    mark_first_rows_responses_as_stale(dataset_name=dataset)
    splits_queue.add_job(dataset=dataset, force=force)


def update_first_rows(dataset: str, config: str, split: str, force: bool = False) -> None:
    logging.debug(f"refresh /first-rows for {dataset}, {config}, {split}")
    mark_first_rows_responses_as_stale(dataset_name=dataset, config_name=config, split_name=split)
    first_rows_queue.add_job(dataset=dataset, config=config, split=split, force=force)
