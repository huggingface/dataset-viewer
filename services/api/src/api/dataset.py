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
parquet_queue = Queue(type=JobType.PARQUET.value)


class LoggedError(Exception):
    def __init__(self, message: str):
        self.message = message
        logging.debug(self.message)
        super().__init__(self.message)


class UnsupportedDatasetError(LoggedError):
    def __init__(self, dataset: str):
        super().__init__(f"Dataset '{dataset}' is not supported (does not exist or is private)")


class SplitsResponseError(LoggedError):
    def __init__(self, dataset: str):
        super().__init__(f"Splits response for dataset '{dataset}' is an error")


class MissingSplitError(LoggedError):
    def __init__(self, dataset: str):
        super().__init__(f"Split does not exist in the /splits response for dataset '{dataset}'")


def check_support(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
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
        `None`
    Raises:
        UnsupportedDatasetError: if the dataset is not supported
    """
    try:
        # note that token is required to access gated dataset info
        info = HfApi(endpoint=hf_endpoint).dataset_info(dataset, token=hf_token)
        if info.private is True:
            raise UnsupportedDatasetError(dataset=dataset)
    except RepositoryNotFoundError as e:
        raise UnsupportedDatasetError(dataset=dataset) from e


def update_dataset(dataset: str, hf_endpoint: str, hf_token: Optional[str] = None, force: bool = False) -> None:
    check_support(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
    logging.debug(f"refresh dataset='{dataset}'")
    splits_queue.add_job(dataset=dataset, force=force)
    parquet_queue.add_job(dataset=dataset, force=force)


def delete_dataset(dataset: str) -> None:
    logging.debug(f"delete cache for dataset='{dataset}'")
    delete_dataset_responses(dataset=dataset)


def move_dataset(
    from_dataset: str, to_dataset: str, hf_endpoint: str, hf_token: Optional[str] = None, force: bool = False
) -> None:
    logging.debug(f"move dataset '{from_dataset}' to '{to_dataset}'")
    update_dataset(dataset=to_dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=force)
    # ^ can raise UnsupportedDatasetError
    delete_dataset(dataset=from_dataset)


def check_splits_in_process(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
    if splits_queue.is_job_in_process(dataset=dataset):
        # the /splits response is not ready yet
        return
    update_dataset(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=False)


def check_parquet_in_process(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
    if parquet_queue.is_job_in_process(dataset=dataset):
        # the /parquet response is not ready yet
        return
    update_dataset(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=False)


def check_first_rows_in_process(
    dataset: str, config: str, split: str, hf_endpoint: str, hf_token: Optional[str] = None
) -> None:
    if first_rows_queue.is_job_in_process(
        dataset=dataset, config=config, split=split
    ) or splits_queue.is_job_in_process(dataset=dataset):
        return

    # a bit convoluted, but to check if the first-rows response should exist,
    # we have to check the content of the /splits response for the same dataset
    try:
        result = get_response(kind=CacheKind.SPLITS.value, dataset=dataset)
    except DoesNotExist:
        # the splits responses does not exist, update
        update_dataset(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
        return

    if result["http_status"] != HTTPStatus.OK:
        # the /splits response is an error, the first rows response should not exist
        raise SplitsResponseError(dataset=dataset)

    if all(
        split_item["dataset"] != dataset or split_item["config"] != config or split_item["split"] != split
        for split_item in result["content"]["splits"]
    ):
        # the split is not listed in the /splits response, so it's normal that it's not in the cache
        raise MissingSplitError(dataset=dataset)

    # The split is listed in the /splits response.
    # Let's refresh *the whole dataset*, because something did not work
    # Note that we "force" the refresh
    #
    # Caveat: we don't even check if the /first-rows response already exists in the cache,
    # because we assume it's the reason why one would call this function
    update_dataset(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, force=True)
