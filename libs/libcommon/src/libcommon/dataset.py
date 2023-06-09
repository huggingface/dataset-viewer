# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from huggingface_hub.hf_api import DatasetInfo, HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError, RevisionNotFoundError

from libcommon.exceptions import (
    CustomError,
    DatasetInfoHubRequestError,
    DatasetNotFoundError,
    DatasetRevisionEmptyError,
    DatasetRevisionNotFoundError,
    DisabledViewerError,
)

DOES_NOT_EXIST_OR_PRIVATE_DATASET_ERROR_MESSAGE = (
    "The dataset does not exist on the Hub, or is private. Private datasets are not yet supported."
)


def raise_if_not_supported(dataset_info: DatasetInfo) -> None:
    """
    Raise an error if the dataset is not supported by the datasets-server.
    Args:
        dataset_info (`DatasetInfo`):
            The dataset info.
    Returns:
        `None`
    Raises the following errors:
        - [`~exceptions.DatasetNotFoundError`]
          if the dataset id does not exist, or if the dataset is private
        - [`~exceptions.DisabledViewerError`]
          if the dataset viewer is disabled.
    """
    if not dataset_info.id or dataset_info.private:
        raise DatasetNotFoundError(DOES_NOT_EXIST_OR_PRIVATE_DATASET_ERROR_MESSAGE)
    if dataset_info.cardData and not dataset_info.cardData.get("viewer", True):
        raise DisabledViewerError("The dataset viewer has been disabled on this dataset.")


def is_supported(dataset_info: DatasetInfo) -> bool:
    """
    Check if the dataset is supported by the datasets-server.
    Args:
        dataset_info (`DatasetInfo`):
            The dataset info.
    Returns:
        `bool`: True if the dataset is supported, False otherwise.
    """
    try:
        raise_if_not_supported(dataset_info)
    except CustomError:
        return False
    return True


def get_dataset_info_for_supported_datasets(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    revision: Optional[str] = None,
    files_metadata: bool = False,
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
        revision (`str`, *optional*, defaults to None):
            The revision of the dataset repository from which to get the
            information.
        files_metadata (`bool`, *optional*, defaults to False):
            Whether or not to retrieve metadata for files in the repository
            (size, LFS metadata, etc). Defaults to `False`.
    Returns:
        `DatasetInfo`: the dataset info.
    Raises the following errors:
        - [`~exceptions.DatasetInfoHubRequestError`]
          if the request to the Hub to get the dataset info failed or timed out.
        - [`~exceptions.DatasetNotFoundError`]:
          if the dataset does not exist, or if the token does not give the sufficient access to the dataset,
          or if the dataset is private (private datasets are not supported by the datasets server).
        - [`~exceptions.DatasetRevisionNotFoundError`]
          if the git revision (branch, commit) does not exist in the repository.
        - [`~exceptions.DisabledViewerError`]
          if the dataset viewer is disabled.
        - ['requests.exceptions.HTTPError'](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
          any other error when asking access
    """
    try:
        dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(
            repo_id=dataset,
            token=hf_token,
            timeout=hf_timeout_seconds,
            revision=revision,
            files_metadata=files_metadata,
        )
    except CustomError as err:
        raise err
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError(DOES_NOT_EXIST_OR_PRIVATE_DATASET_ERROR_MESSAGE, cause=err) from err
    except RevisionNotFoundError as err:
        raise DatasetRevisionNotFoundError(
            f"The default branch cannot be found in dataset {dataset} on the Hub.", cause=err
        ) from err
    except Exception as err:
        raise DatasetInfoHubRequestError(
            (
                "Request to the Hub to get the dataset info failed or timed out. Please try again later, it's a"
                " temporary internal issue."
            ),
            cause=err,
        ) from err
    raise_if_not_supported(dataset_info)
    return dataset_info


def get_dataset_git_revision(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> str:
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
    Raises the following errors:
        - [`~exceptions.DatasetInfoHubRequestError`]
          if the request to the Hub to get the dataset info failed or timed out.
        - [`~exceptions.DatasetNotFoundError`]:
          if the dataset does not exist, or if the token does not give the sufficient access to the dataset,
          or if the dataset is private (private datasets are not supported by the datasets server).
        - [`~exceptions.DatasetRevisionEmptyError`]
          if the current git revision (branch, commit) could not be obtained.
        - [`~exceptions.DatasetRevisionNotFoundError`]
          if the git revision (branch, commit) does not exist in the repository.
        - [`~exceptions.DisabledViewerError`]
          if the dataset viewer is disabled.
        - ['requests.exceptions.HTTPError'](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
          any other error when asking access
    """
    sha = get_dataset_info_for_supported_datasets(
        dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
    ).sha
    if sha is None:
        raise DatasetRevisionEmptyError(f"The dataset {dataset} has no git revision.")
    return sha  # type: ignore


def get_supported_dataset_infos(hf_endpoint: str, hf_token: Optional[str] = None) -> list[DatasetInfo]:
    return [d for d in HfApi(endpoint=hf_endpoint, token=hf_token).list_datasets() if is_supported(d)]
