# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from enum import Enum
from http import HTTPStatus
from typing import Optional

from huggingface_hub.hf_api import HfApi  # type: ignore
from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore
from libcache.simple_cache import (
    DoesNotExist,
    delete_first_rows_responses,
    delete_splits_responses,
    get_splits_response,
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
)
from libqueue.queue import add_job, is_job_in_process

logger = logging.getLogger(__name__)


class JobType(Enum):
    SPLITS = "/splits"
    FIRST_ROWS = "/first-rows"


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


def update(dataset: str) -> None:
    logger.debug(f"webhook: refresh {dataset}")
    mark_splits_responses_as_stale(dataset)
    mark_first_rows_responses_as_stale(dataset)
    add_job(type=JobType.SPLITS.value, dataset=dataset)


def delete(dataset: str) -> None:
    logger.debug(f"webhook: delete {dataset}")
    delete_splits_responses(dataset)
    delete_first_rows_responses(dataset)


def is_splits_in_process(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> bool:
    if is_job_in_process(type=JobType.SPLITS.value, dataset=dataset):
        return True
    if is_supported(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token):
        update(dataset=dataset)
        return True
    return False


def is_first_rows_in_process(
    dataset: str, config: str, split: str, hf_endpoint: str, hf_token: Optional[str] = None
) -> bool:
    if is_job_in_process(type=JobType.FIRST_ROWS.value, dataset=dataset, config=config, split=split):
        return True

    # a bit convoluted, but checking if the first-rows response should exist
    # requires to first parse the /splits response for the same dataset
    if is_job_in_process(type=JobType.SPLITS.value, dataset=dataset):
        return True
    try:
        response, http_status, _ = get_splits_response(dataset)
        if http_status == HTTPStatus.OK and any(
            split_item["dataset"] == dataset or split_item["config"] == config or split_item["split"] == split
            for split_item in response["splits"]
        ):
            # The splits is listed in the /splits response.
            # Let's refresh *the whole dataset*, because something did not work
            #
            # Caveat: we don't check if the /first-rows response already exists in the cache,
            # because we assume it's the reason why one would call this function
            update(dataset=dataset)
            return True
    except DoesNotExist:
        # the splits responses does not exist, let's check if it should
        return is_splits_in_process(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
    return False
