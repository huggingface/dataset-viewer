# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from typing import Optional

from huggingface_hub.hf_api import DatasetInfo, HfApi

from libcommon.exceptions import DatasetInBlockListError
from libcommon.orchestrator import remove_dataset, set_revision
from libcommon.storage_client import StorageClient
from libcommon.utils import Priority, SupportStatus, raise_if_blocked


@dataclass
class DatasetStatus:
    dataset: str
    revision: str
    support_status: SupportStatus


def is_blocked(
    dataset: str,
    blocked_datasets: Optional[list[str]] = None,
) -> bool:
    if not blocked_datasets:
        logging.debug("No datasets block list.")
        return False
    try:
        raise_if_blocked(dataset, blocked_datasets)
        logging.debug(f"Dataset is not blocked. Block list is: {blocked_datasets}, dataset is: {dataset}")
        return False
    except DatasetInBlockListError:
        logging.debug(f"Dataset is blocked. Block list is: {blocked_datasets}, dataset is: {dataset}")
        return True


def get_support_status(
    dataset_info: DatasetInfo,
    blocked_datasets: Optional[list[str]] = None,
) -> SupportStatus:
    """
      blocked_datasets (list[str]): The list of blocked datasets. Supports Unix shell-style wildcards in the dataset
    name, e.g. "open-llm-leaderboard/*" to block all the datasets in the `open-llm-leaderboard` namespace. They
    are not allowed in the namespace name.
    """
    return (
        SupportStatus.UNSUPPORTED
        if (
            is_blocked(dataset_info.id, blocked_datasets)
            # ^ TODO: should we check here?
            # As we don't keep the track of unsupported datasets,
            # we lose the ability to send a meaningful error message to the user, as previously done:
            #   "This dataset has been disabled for now. Please open an issue in"
            #   " https://github.com/huggingface/datasets-server if you want this dataset to be supported."
            # A solution is to double-check if the dataset is blocked in the API routes.
            or dataset_info.disabled
            or (dataset_info.cardData and not dataset_info.cardData.get("viewer", True))
            or dataset_info.private
        )
        else SupportStatus.PUBLIC
    )


def get_dataset_status(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    blocked_datasets: Optional[list[str]] = None,
) -> DatasetStatus:
    # let's the exceptions bubble up if any
    dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(
        repo_id=dataset, token=hf_token, timeout=hf_timeout_seconds, files_metadata=False
    )
    revision = dataset_info.sha
    if not revision:
        raise ValueError(f"Cannot get the git revision of dataset {dataset}. Let's do nothing.")
    support_status = get_support_status(
        dataset_info=dataset_info,
        blocked_datasets=blocked_datasets,
    )
    return DatasetStatus(dataset=dataset, revision=revision, support_status=support_status)


def delete_dataset(dataset: str, storage_clients: Optional[list[StorageClient]] = None) -> None:
    """
    Delete a dataset

    Args:
        dataset (str): the dataset

    Returns: None.
    """
    logging.debug(f"delete cache for dataset='{dataset}'")
    remove_dataset(dataset=dataset, storage_clients=storage_clients)


def update_dataset(
    dataset: str,
    cache_max_days: int,
    hf_endpoint: str,
    blocked_datasets: Optional[list[str]] = None,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    priority: Priority = Priority.LOW,
    error_codes_to_retry: Optional[list[str]] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> bool:
    # let's the exceptions bubble up if any
    dataset_status = get_dataset_status(
        dataset=dataset,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        blocked_datasets=blocked_datasets,
    )
    if dataset_status.support_status == SupportStatus.UNSUPPORTED:
        logging.warning(f"Dataset {dataset} is not supported. Let's delete the dataset.")
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
        return False
    set_revision(
        dataset=dataset,
        revision=dataset_status.revision,
        priority=priority,
        error_codes_to_retry=error_codes_to_retry,
        cache_max_days=cache_max_days,
    )
    return True
