# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from libcommon.exceptions import (
    NotSupportedDisabledRepositoryError,
    NotSupportedDisabledViewerError,
    NotSupportedError,
    NotSupportedPrivateRepositoryError,
    NotSupportedRepositoryNotFoundError,
)
from libcommon.orchestrator import get_revision, remove_dataset, set_revision
from libcommon.storage_client import StorageClient
from libcommon.utils import Priority, raise_if_blocked


def get_dataset_info(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> DatasetInfo:
    # let's the exceptions bubble up if any
    return HfApi(endpoint=hf_endpoint).dataset_info(
        repo_id=dataset, token=hf_token, timeout=hf_timeout_seconds, files_metadata=False
    )


class DisabledRepoError(HfHubHTTPError):
    pass


def get_latest_dataset_revision_if_supported_or_raise(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    blocked_datasets: Optional[list[str]] = None,
) -> str:
    try:
        dataset_info = get_dataset_info(
            dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
        )
    except RepositoryNotFoundError as e:
        raise NotSupportedRepositoryNotFoundError(f"Repository {dataset} is not found.", e) from e
    except HfHubHTTPError as e:
        response = e.response
        if response.headers.get("X-Error-Message") == "Access to this resource is disabled.":
            # ^ huggingface_hub does not yet provide a specific exception for this error
            # Let's catch DisabledRepoError instead once https://github.com/huggingface/huggingface_hub/pull/1965
            # is released.
            raise NotSupportedDisabledRepositoryError(f"Repository {dataset} is disabled.", e) from e
        raise
    revision = dataset_info.sha
    if not revision:
        raise ValueError(f"Cannot get the git revision of dataset {dataset}.")
    if dataset_info.disabled:
        # ^ in most cases, get_dataset_info should already have raised. Anyway, we double-check here.
        raise NotSupportedDisabledRepositoryError(f"Not supported: dataset repository {dataset} is disabled.")
    if dataset_info.private:
        raise NotSupportedPrivateRepositoryError(f"Not supported: dataset repository {dataset} is private.")
    if dataset_info.cardData and not dataset_info.cardData.get("viewer", True):
        raise NotSupportedDisabledViewerError(f"Not supported: dataset viewer is disabled in {dataset} configuration.")
    if blocked_datasets:
        raise_if_blocked(dataset=dataset, blocked_datasets=blocked_datasets)
    return str(revision)


def get_current_revision(
    dataset: str,
) -> Optional[str]:
    logging.debug(f"get current revision for dataset='{dataset}'")
    return get_revision(dataset=dataset)


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
    hf_endpoint: str,
    blocked_datasets: Optional[list[str]] = None,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    priority: Priority = Priority.LOW,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    """
      blocked_datasets (list[str]): The list of blocked datasets. Supports Unix shell-style wildcards in the dataset
    name, e.g. "open-llm-leaderboard/*" to block all the datasets in the `open-llm-leaderboard` namespace. They
    are not allowed in the namespace name.
    """
    # let's the exceptions bubble up if any
    try:
        revision = get_latest_dataset_revision_if_supported_or_raise(
            dataset=dataset,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            hf_timeout_seconds=hf_timeout_seconds,
            blocked_datasets=blocked_datasets,
        )
    except NotSupportedError as e:
        logging.warning(f"Dataset {dataset} is not supported ({type(e)}). Let's delete the dataset.")
        delete_dataset(dataset=dataset, storage_clients=storage_clients)
        raise
    set_revision(
        dataset=dataset,
        revision=revision,
        priority=priority,
    )
