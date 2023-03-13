# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Literal, Optional

import requests
from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.utils._headers import build_hf_headers

from libcommon.exceptions import CustomError

DatasetErrorCode = Literal[
    "AskAccessHubRequestError",
    "DatasetInfoHubRequestError",
    "DatasetNotFoundError",
    "GatedDisabledError",
    "GatedExtraFieldsError",
]


class DatasetError(CustomError):
    """Base class for dataset exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: DatasetErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=str(code), cause=cause, disclose_cause=disclose_cause
        )


class AskAccessHubRequestError(DatasetError):
    """Raised when the request to the Hub's ask-access endpoint times out."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="AskAccessHubRequestError",
            cause=cause,
            disclose_cause=False,
        )


class DatasetInfoHubRequestError(DatasetError):
    """Raised when the request to the Hub's dataset-info endpoint times out."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="DatasetInfoHubRequestError",
            cause=cause,
            disclose_cause=False,
        )


class DatasetNotFoundError(DatasetError):
    """Raised when the dataset does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="DatasetNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class GatedDisabledError(DatasetError):
    """Raised when the dataset is gated, but disabled."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="GatedDisabledError",
            cause=cause,
            disclose_cause=False,
        )


class GatedExtraFieldsError(DatasetError):
    """Raised when the dataset is gated, with extra fields."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="GatedExtraFieldsError",
            cause=cause,
            disclose_cause=False,
        )


def ask_access(
    dataset: str, hf_endpoint: str, hf_token: Optional[str], hf_timeout_seconds: Optional[float] = None
) -> None:
    """
    Ask access to the dataset repository.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        hf_timeout_seconds (`float`, *optional*, defaults to None):
            The timeout in seconds for the request to the Hub.
    Returns:
        `None`
    Raises:
        - [`~libcommon.dataset.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
    """
    path = f"{hf_endpoint}/datasets/{dataset}/ask-access"
    try:
        r = requests.post(path, headers=build_hf_headers(token=hf_token), timeout=hf_timeout_seconds)
    except Exception as err:
        raise AskAccessHubRequestError(
            (
                "Request to the Hub to get access to the dataset failed or timed out. Please try again later, it's a"
                " temporary internal issue."
            ),
            err,
        ) from err
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if r.status_code == 400:
            if r.headers and r.headers.get("X-Error-Code") == "RepoNotGated":
                return  # the dataset is not gated
            raise GatedExtraFieldsError(
                "The dataset is gated with extra fields: not supported at the moment."
            ) from err
        if r.status_code == 403:
            raise GatedDisabledError("The dataset is gated and access is disabled.") from err
        if r.status_code in [401, 404]:
            raise DatasetNotFoundError(
                "The dataset does not exist on the Hub, or is private. Private datasets are not yet supported."
            ) from err
        raise err


def get_dataset_info_for_supported_datasets(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    allow_recursive_call: bool = True,
) -> DatasetInfo:
    """
    Get the DatasetInfo of the dataset, after checking if it's supported (no private datasets).
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        hf_timeout_seconds (`float`, *optional*, defaults to None):
            The timeout in seconds for the request to the Hub.
        allow_recursive_call (`bool`, *optional*, defaults to True):
            If True, the function will be called again in case of success after asking for access
    Returns:
        `DatasetInfo`: the dataset info.
    <Tip>
    Raises the following errors:
        - [`~libcommon.dataset.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~libcommon.dataset.DatasetInfoHubRequestError`]: if the request to the Hub to get the dataset
            info failed or timed out.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server), or if the default branch cannot
            be found in the dataset.
        - ['~requests.exceptions.HTTPError']: any other error when asking access
    </Tip>
    """
    try:
        dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(
            repo_id=dataset, token=hf_token, timeout=hf_timeout_seconds
        )
    except RepositoryNotFoundError:
        if allow_recursive_call:
            ask_access(
                dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
            )
            return get_dataset_info_for_supported_datasets(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                hf_timeout_seconds=hf_timeout_seconds,
                allow_recursive_call=False,
            )
    except RevisionNotFoundError as err:
        raise DatasetNotFoundError(
            f"The default branch cannot be found in dataset {dataset} on the Hub.", cause=err
        ) from err
    except Exception as err:
        raise DatasetInfoHubRequestError(
            (
                "Request to the Hub to get the dataset info failed or timed out. Please try again later, it's a"
                " temporary internal issue."
            ),
            err,
        ) from err
    if dataset_info.private:
        raise DatasetNotFoundError("The dataset is private and private datasets are not yet supported.")
    return dataset_info


def get_dataset_git_revision(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Optional[str]:
    """
    Get the git revision of the dataset.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        hf_timeout_seconds (`float`, *optional*, defaults to None):
            The timeout in seconds for the request to the Hub.
    Returns:
        `Union[str, None]`: the dataset git revision (sha) if any.
    <Tip>
    Raises the following errors:
        - [`~libcommon.dataset.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~libcommon.dataset.DatasetInfoHubRequestError`]: if the request to the Hub to get the dataset
            info failed or timed out.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
    </Tip>
    """
    return get_dataset_info_for_supported_datasets(
        dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
    ).sha


def check_support(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
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
        hf_timeout_seconds (`float`, *optional*, defaults to None):
            The timeout in seconds for the request to the Hub.
    Returns:
        `None`
    Raises:
        - [`~libcommon.dataset.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~libcommon.dataset.DatasetInfoHubRequestError`]: if the request to the Hub to get the dataset
            info failed or timed out.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
    """
    get_dataset_info_for_supported_datasets(
        dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
    )


def get_supported_datasets(hf_endpoint: str, hf_token: Optional[str] = None) -> list[str]:
    return [d.id for d in HfApi(endpoint=hf_endpoint, token=hf_token).list_datasets() if d.id and not d.private]
    # no timeout on this function. It's used only in the admin service
