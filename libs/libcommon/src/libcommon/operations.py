# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from typing import Optional, Union

from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils import get_session, hf_raise_for_status, raise_if_blocked, validate_hf_hub_args

from libcommon.exceptions import DatasetInBlockListError
from libcommon.orchestrator import DatasetOrchestrator
from libcommon.utils import Priority, SupportStatus


@dataclass
class DatasetStatus:
    dataset: str
    revision: str
    support_status: SupportStatus


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
        entity: str,
        *,
        timeout: Optional[float] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> EntityInfo:
        """
        Get information on an entity on huggingface.co.

        You have to pass an acceptable token.

        Args:
            entity (`str`):
                An entity (user or an organization).
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
        path = f"{self.endpoint}/api/whoisthis/{entity}"

        r = get_session().get(path, headers=headers, timeout=timeout)
        hf_raise_for_status(r)
        data = r.json()
        return EntityInfo(**data)


def is_blocked(
    dataset: str,
    blocked_datasets: list[str],
) -> None:
    try:
        raise_if_blocked(dataset, blocked_datasets)
        return False
    except DatasetInBlockListError:
        return True


def get_support_status(
    dataset_info: DatasetInfo,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    blocked_datasets: list[str] = list,
) -> SupportStatus:
    """
      blocked_datasets (list[str]): The list of blocked datasets. Supports Unix shell-style wildcards in the dataset
    name, e.g. "open-llm-leaderboard/*" to block all the datasets in the `open-llm-leaderboard` namespace. They
    are not allowed in the namespace name.
    """
    if (
        is_blocked(dataset_info.id, blocked_datasets)
        or dataset_info.disabled
        or (dataset_info.cardData and not dataset_info.cardData.get("viewer", True))
    ):
        return SupportStatus.UNSUPPORTED
    if not dataset_info.private:
        return SupportStatus.PUBLIC
    author = dataset_info.author
    if not author:
        return SupportStatus.UNSUPPORTED
    entity_info = CustomHfApi(endpoint=hf_endpoint).whoisthis(
        entity=author,
        token=hf_token,
        timeout=hf_timeout_seconds,
    )
    if entity_info["is_pro"]:
        return SupportStatus.PRO_USER
    if entity_info["is_enterprise"]:
        return SupportStatus.ENTERPRISE_ORG
    return SupportStatus.UNSUPPORTED


def get_dataset_status(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    revision: Optional[str] = None,
    blocked_datasets: list[str] = list,
) -> DatasetStatus:
    # let's the exceptions bubble up if any
    dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(
        repo_id=dataset, token=hf_token, timeout=hf_timeout_seconds, files_metadata=False, revision=revision
    )
    revision = dataset_info.sha
    if not revision:
        raise ValueError(f"Cannot get the git revision of dataset {dataset}. Let's do nothing.")
    support_status = get_support_status(
        dataset_info=dataset_info,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        blocked_datasets=blocked_datasets,
    )
    return DatasetStatus(dataset=dataset, revision=revision, support_status=support_status)


def backfill_dataset(
    dataset: str,
    revision: str,
    cache_max_days: int,
    priority: Priority = Priority.LOW,
) -> None:
    """
    Update a dataset

    Args:
        dataset (str): the dataset
        revision (str): The revision of the dataset.
        cache_max_days (int): the number of days to keep the cache
        priority (Priority, optional): The priority of the job. Defaults to Priority.LOW.

    Returns: None.

    Raises the following errors:
        - [`libcommon.exceptions.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
    """
    logging.debug(f"backfill {dataset=} {revision=} {priority=}")
    DatasetOrchestrator(dataset=dataset).set_revision(
        revision=revision, priority=priority, error_codes_to_retry=[], cache_max_days=cache_max_days
    )


def delete_dataset(dataset: str) -> None:
    """
    Delete a dataset

    Args:
        dataset (str): the dataset

    Returns: None.
    """
    logging.debug(f"delete cache for dataset='{dataset}'")
    DatasetOrchestrator(dataset=dataset).remove_dataset()


def check_support_and_act(
    dataset: str,
    cache_max_days: int,
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    priority: Priority = Priority.LOW,
    revision: Optional[str] = None,
) -> bool:
    # let's the exceptions bubble up if any
    dataset_status = get_dataset_status(
        dataset=dataset,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        revision=revision,
        blocked_datasets=blocked_datasets,
    )
    if dataset_status.support_status == SupportStatus.UNSUPPORTED:
        logging.warning(f"Dataset {dataset} is not supported. Let's delete the dataset.")
        delete_dataset(dataset=dataset)
        return False
    backfill_dataset(
        dataset=dataset,
        revision=dataset_status.revision,
        cache_max_days=cache_max_days,
        priority=priority,
    )
    return True
