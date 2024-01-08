# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from huggingface_hub.hf_api import HfApi

from libcommon.exceptions import DatasetInBlockListError, DatasetNotFoundError
from libcommon.orchestrator import remove_dataset, set_revision
from libcommon.storage_client import StorageClient
from libcommon.utils import Priority, raise_if_blocked


class NotSupportedError(Exception):
    pass


def get_dataset_revision_if_supported_or_raise(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> str:
    # let's the exceptions bubble up if any
    dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(
        repo_id=dataset, token=hf_token, timeout=hf_timeout_seconds, files_metadata=False
    )
    revision = dataset_info.sha
    if not revision:
        raise NotSupportedError(f"Cannot get the git revision of dataset {dataset}. Let's do nothing.")
    if (
        dataset_info.disabled
        or (dataset_info.cardData and not dataset_info.cardData.get("viewer", True))
        or dataset_info.private
    ):
        raise NotSupportedError(f"Dataset {dataset} is not supported. Let's do nothing.")
    return str(revision)


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
    let_raise_if_blocked: Optional[bool] = False,
) -> None:
    """
      blocked_datasets (list[str]): The list of blocked datasets. Supports Unix shell-style wildcards in the dataset
    name, e.g. "open-llm-leaderboard/*" to block all the datasets in the `open-llm-leaderboard` namespace. They
    are not allowed in the namespace name.
    """
    # let's the exceptions bubble up if any
    try:
        revision = get_dataset_revision_if_supported_or_raise(
            dataset=dataset,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            hf_timeout_seconds=hf_timeout_seconds,
        )
        # we check if the dataset is blocked at last, to avoid disclosing the existence of a private dataset, if any is blocked
        if blocked_datasets:
            raise_if_blocked(dataset, blocked_datasets)
    except (NotSupportedError, DatasetInBlockListError) as e:
        logging.warning(f"Dataset {dataset} is not supported. Let's delete the dataset.")
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
        if isinstance(e, DatasetInBlockListError) and let_raise_if_blocked:
            raise
        raise DatasetNotFoundError(f"Dataset {dataset} does not exist or is not supported.")
    set_revision(
        dataset=dataset,
        revision=revision,
        priority=priority,
        error_codes_to_retry=error_codes_to_retry,
        cache_max_days=cache_max_days,
    )
