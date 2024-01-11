# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from typing import Optional, Union

from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    get_session,
    hf_raise_for_status,
    validate_hf_hub_args,
)

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


@dataclass
class EntityInfo:
    """
    Contains (very partial) information about an entity on the Hub.

    <Tip>

    Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made.

    </Tip>

    Attributes:
        is_pro (`bool`, *optional*):
            Is the entity a pro user.
        is_enterprise (`bool`, *optional*):
            Is the entity an enterprise organization.
    """

    is_pro: Optional[bool]
    is_enterprise: Optional[bool]

    def __init__(self, **kwargs) -> None:  # type: ignore
        self.is_pro = kwargs.pop("isPro", None)
        self.is_enterprise = kwargs.pop("isEnterprise", None)


class CustomHfApi(HfApi):  # type: ignore
    @validate_hf_hub_args  # type: ignore
    def whoisthis(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> EntityInfo:
        """
        Get information on an entity on huggingface.co.

        You have to pass an acceptable token.

        Args:
            name (`str`):
                Name of a user or an organization.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`hf_api.EntityInfo`]: The entity information.
        """
        headers = self._build_hf_headers(token=token)
        path = f"{self.endpoint}/api/whoisthis"
        params = {"name": name}

        r = get_session().get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        data = r.json()
        return EntityInfo(**data)


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


def get_entity_info(
    author: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> EntityInfo:
    # let's the exceptions bubble up if any
    return CustomHfApi(endpoint=hf_endpoint).whoisthis(  # type: ignore
        name=author,
        token=hf_token,
        timeout=hf_timeout_seconds,
    )


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
        author = dataset_info.author
        if not author:
            raise ValueError(f"Cannot get the author of dataset {dataset}.")
        entity_info = get_entity_info(
            author=author, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
        )
        if (not entity_info.is_pro) and (not entity_info.is_enterprise):
            raise NotSupportedPrivateRepositoryError(
                f"Not supported: dataset repository {dataset} is private. Private datasets are only supported for pro users and enterprise organizations."
            )
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
