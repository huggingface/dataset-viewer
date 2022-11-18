# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from libcache.simple_cache import DoesNotExist, delete_dataset_responses, get_response
from libqueue.queue import Queue

from api.utils import CacheKind, JobType

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


def update(dataset: str, hf_endpoint: str, hf_token: Optional[str] = None, force: bool = False) -> bool:
    if is_supported(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token):
        logging.debug(f"refresh dataset='{dataset}'")
        splits_queue.add_job(dataset=dataset, force=force)
        return True
    else:
        logging.debug(f"can't refresh dataset='{dataset}', it's not supported (does not exist, private, etc.)")
        return False


def delete(dataset: str) -> bool:
    logging.debug(f"delete cache for dataset='{dataset}'")
    delete_dataset_responses(dataset=dataset)
    return True


def move(
    from_dataset: str, to_dataset: str, hf_endpoint: str, hf_token: Optional[str] = None, force: bool = False
) -> bool:
    # not optimal as we might try to rename instead
    if update(dataset=to_dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=force):
        return delete(dataset=from_dataset)
    else:
        return False


def is_splits_in_process(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> bool:
    if splits_queue.is_job_in_process(dataset=dataset):
        # the /splits response is not ready yet
        return True
    return update(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=False)


def is_first_rows_in_process(
    dataset: str, config: str, split: str, hf_endpoint: str, hf_token: Optional[str] = None
) -> bool:
    if first_rows_queue.is_job_in_process(
        dataset=dataset, config=config, split=split
    ) or splits_queue.is_job_in_process(dataset=dataset):
        return True

    # a bit convoluted, but to check if the first-rows response should exist,
    # we have to check the content of the /splits response for the same dataset
    try:
        result = get_response(kind=CacheKind.SPLITS.value, dataset=dataset)
    except DoesNotExist:
        # the splits responses does not exist, update
        return update(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)

    if result["http_status"] == HTTPStatus.OK and any(
        split_item["dataset"] == dataset or split_item["config"] == config or split_item["split"] == split
        for split_item in result["content"]["splits"]
    ):
        # The split is listed in the /splits response.
        # Let's refresh *the whole dataset*, because something did not work
        # Note that we "force" the refresh
        #
        # Caveat: we don't check if the /first-rows response already exists in the cache,
        # because we assume it's the reason why one would call this function
        return update(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=True)
    else:
        # the /splits response is an error, or the split is not listed in the /splits response, so it's normal
        # that it's not in the cache
        return False
