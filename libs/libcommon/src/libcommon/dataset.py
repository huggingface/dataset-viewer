# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Literal, Optional

import requests
from huggingface_hub.hf_api import (
    DatasetInfo,
    HfApi,
    RepositoryNotFoundError,
    build_hf_headers,
)

from libcommon.exceptions import CustomError

DatasetErrorCode = Literal[
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


def ask_access(dataset: str, hf_endpoint: str, hf_token: Optional[str]) -> None:
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
    Returns:
        `None`
    Raises:
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
    r = requests.post(path, headers=build_hf_headers(token=hf_token))
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if r.status_code == 400:
            raise GatedExtraFieldsError(
                "The dataset is gated with extra fields: not supported at the moment."
            ) from err
        if r.status_code == 403:
            raise GatedDisabledError("The dataset is gated and access is disabled.") from err
        if r.status_code in [401, 404]:
            raise DatasetNotFoundError("The dataset does not exist on the Hub, or is private.") from err
        raise err


def get_dataset_info_for_supported_datasets(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
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
    Returns:
        `DatasetInfo`: the dataset info.
    <Tip>
    Raises the following errors:
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
    try:
        dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(repo_id=dataset, token=hf_token)
    except RepositoryNotFoundError:
        ask_access(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
    if dataset_info.private is True:
        raise DatasetNotFoundError("The dataset does not exist on the Hub, or is private.")
    return dataset_info


def get_dataset_git_revision(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
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
    Returns:
        `Union[str, None]`: the dataset git revision (sha) if any.
    <Tip>
    Raises the following errors:
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
    return get_dataset_info_for_supported_datasets(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token).sha


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
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
    """
    get_dataset_info_for_supported_datasets(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)


def get_supported_datasets(hf_endpoint: str, hf_token: Optional[str] = None) -> list[str]:
    return [
        d.id
        for d in HfApi(endpoint=hf_endpoint, token=hf_token).list_datasets()
        if d.id and not d.private
    ]
