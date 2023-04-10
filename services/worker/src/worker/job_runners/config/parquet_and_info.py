# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import glob
import logging
import re
from functools import partial
from http import HTTPStatus
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Set, Tuple, TypedDict
from urllib.parse import quote

import datasets
import datasets.config
import numpy as np
import requests
from datasets import DownloadConfig, get_dataset_config_info, load_dataset_builder
from datasets.builder import DatasetBuilder
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.download import StreamingDownloadManager
from datasets.utils.file_utils import (
    get_authentication_headers_for_url,
    http_head,
    is_relative_path,
    url_or_path_join,
)
from datasets.utils.py_utils import asdict, map_nested
from huggingface_hub._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
)
from huggingface_hub.hf_api import DatasetInfo, HfApi, RepoFile
from huggingface_hub.utils._errors import RepositoryNotFoundError, RevisionNotFoundError
from libcommon.constants import PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION
from libcommon.dataset import DatasetNotFoundError, ask_access
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.config import AppConfig, ParquetAndInfoConfig
from worker.job_runner import CompleteJobResult, JobRunnerError, ParameterMissingError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.job_runners.config_names import ConfigNamesError

ConfigParquetAndInfoJobRunnerErrorCode = Literal[
    "DatasetRevisionNotFoundError",
    "EmptyDatasetError",
    "DatasetInBlockListError",
    "DatasetTooBigFromHubError",
    "DatasetTooBigFromDatasetsError",
    "UnsupportedExternalFilesError",
    "DatasetWithTooManyExternalFilesError",
    "DatasetWithTooBigExternalFilesError",
    "ExternalFilesSizeRequestHTTPError",
    "ExternalFilesSizeRequestConnectionError",
    "ExternalFilesSizeRequestTimeoutError",
    "ExternalFilesSizeRequestError",
    "PreviousStepStatusError",
    "PreviousStepFormatError",
]


class ConfigParquetAndInfoJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigParquetAndInfoJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class DatasetRevisionNotFoundError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the revision of a dataset repository does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "DatasetRevisionNotFoundError", cause, False)


class EmptyDatasetError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class DatasetInBlockListError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the dataset is in the list of blocked datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetInBlockListError", cause, False)


class DatasetTooBigFromHubError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the dataset size (sum of files on the Hub) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromHubError", cause, False)


class DatasetTooBigFromDatasetsError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromDatasetsError", cause, False)


class PreviousStepStatusError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class ParquetFileItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class ConfigParquetAndInfoResponse(TypedDict):
    parquet_files: List[ParquetFileItem]
    dataset_info: Dict[str, Any]


DATASET_TYPE = "dataset"


class ParquetFile:
    def __init__(self, local_file: str, local_dir: str, config: str):
        if not local_file.startswith(local_dir):
            raise ValueError(f"{local_file} is not in {local_dir}")
        self.local_file = local_file
        self.local_dir = local_dir
        self.config = config

    def repo_file(self) -> str:
        return f'{self.config}/{self.local_file.removeprefix(f"{self.local_dir}/")}'


# TODO: use huggingface_hub's hf_hub_url after
# https://github.com/huggingface/huggingface_hub/issues/1082
def hf_hub_url(repo_id: str, filename: str, hf_endpoint: str, revision: str, url_template: str) -> str:
    return (hf_endpoint + url_template) % (repo_id, quote(revision, safe=""), filename)


p = re.compile(r"(?P<builder>[\w-]+?)-(?P<split>\w+(\.\w+)*?)(-[0-9]{5}-of-[0-9]{5})?.parquet")


def parse_repo_filename(filename: str) -> Tuple[str, str]:
    parts = filename.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid filename: {filename}")
    config, fname = parts
    m = p.match(fname)
    if not m:
        raise ValueError(f"Cannot parse {filename}")
    split = m.group("split")
    return config, split


def create_parquet_file_item(
    repo_file: RepoFile,
    dataset: str,
    config: str,
    hf_endpoint: str,
    target_revision: str,
    url_template: str,
) -> ParquetFileItem:
    if repo_file.size is None:
        raise ValueError(f"Cannot get size of {repo_file.rfilename}")
    _, split = parse_repo_filename(repo_file.rfilename)
    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "url": hf_hub_url(
            repo_id=dataset,
            filename=repo_file.rfilename,
            hf_endpoint=hf_endpoint,
            revision=target_revision,
            url_template=url_template,
        ),
        "filename": Path(repo_file.rfilename).name,
        "size": repo_file.size,
    }


def raise_if_blocked(
    dataset: str,
    blocked_datasets: List[str],
) -> None:
    """
    Raise an error if the dataset is in the list of blocked datasets

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        blocked_datasets (`List[str]`):
            The list of blocked datasets. If empty, no dataset is blocked.
    Returns:
        `None`
    <Tip>
    Raises the following errors:
        - [`~job_runners.config.parquet_and_info.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
    </Tip>
    """
    if dataset in blocked_datasets:
        raise DatasetInBlockListError(
            "The parquet conversion has been disabled for this dataset for now. Please open an issue in"
            " https://github.com/huggingface/datasets-server if you want this dataset to be supported."
        )


def get_dataset_info_or_raise(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    revision: str,
) -> DatasetInfo:
    """
    Return the dataset info if possible.
    Raise an error if the dataset cannot be accessed (does not exist, gated with extra fields, private)

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, `optional`):
            An app authentication token with read access to all the datasets.
        revision (`str`):
            The git revision (e.g. "main" or sha) of the dataset
    Returns:
        `DatasetInfo`: The dataset info
    <Tip>
    Raises the following errors:
        - [`~.job_runner.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~job_runners.config.parquet_and_info.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
    </Tip>
    """
    try:
        dataset_info = HfApi(endpoint=hf_endpoint, token=hf_token).dataset_info(
            repo_id=dataset, revision=revision, files_metadata=True
        )
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    except RevisionNotFoundError as err:
        raise DatasetRevisionNotFoundError("The dataset revision does not exist on the Hub.") from err
    if dataset_info.private:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.")
    return dataset_info


def raise_if_too_big_from_hub(
    dataset_info: DatasetInfo,
    max_dataset_size: int,
) -> None:
    """
    Raise an error if the dataset is too big to be converted to parquet

    Args:
        dataset_info (`DatasetInfo`):
            The dataset info
        max_dataset_size (`int`):
            The maximum size of the dataset in bytes
    Returns:
        `None`
    <Tip>
    Raises the following errors:
        - [`~job_runners.config.parquet_and_info.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
    </Tip>
    """
    dataset_size: int = sum(sibling.size for sibling in dataset_info.siblings if sibling.size is not None)
    if dataset_size > max_dataset_size:
        raise DatasetTooBigFromHubError(
            f"The conversion to parquet is limited to datasets under {max_dataset_size} bytes. "
            f"Current size of files on the hub is {dataset_size} bytes."
        )


def raise_if_too_big_from_datasets(
    dataset: str,
    config: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    revision: str,
    max_dataset_size: int,
) -> None:
    """
    Raise an error if the dataset is too big to be converted to parquet, as measured by the sum of the configs
    sizes given by the datasets library

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            Dataset configuration name
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, `optional`):
            An app authentication token with read access to all the datasets.
        revision (`str`):
            The git revision (e.g. "main" or sha) of the dataset
        max_dataset_size (`int`):
            The maximum size of the dataset in bytes
    Returns:
        `None`
    <Tip>
    Raises the following errors:
        - [`ValueError`]
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~job_runners.config.parquet_and_info.DatasetTooBigFromDatasetsError`]
            If the dataset is too big to be converted to parquet
    </Tip>
    """
    if datasets.config.HF_ENDPOINT != hf_endpoint:
        raise ValueError(
            f"Invalid datasets.config.HF_ENDPOINT value: '{datasets.config.HF_ENDPOINT}'. Please set it to:"
            f" '{hf_endpoint}'."
        )
    dataset_size = 0
    with contextlib.suppress(Exception):
        info = get_dataset_config_info(path=dataset, config_name=config, revision=revision, use_auth_token=hf_token)
        dataset_size = info.dataset_size if info.dataset_size is not None else 0
    if dataset_size > max_dataset_size:
        raise DatasetTooBigFromDatasetsError(
            f"The dataset is too big to be converted to Parquet. The size of the dataset ({dataset_size} B, as given"
            f" per the datasets library) exceeds the maximum supported size ({max_dataset_size} B). Please report the"
            " issue."
        )


def raise_if_not_supported(
    dataset: str,
    config: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    committer_hf_token: Optional[str],
    revision: str,
    supported_datasets: List[str],
    blocked_datasets: List[str],
    max_dataset_size: int,
) -> None:
    """
    Raise an error if the dataset is not supported:
    - if the dataset is in the list of blocked datasets
    - if the dataset cannot be accessed (does not exist, gated with extra fields, private)
    - if the dataset is too big, and not in the list of supported datasets

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            Dataset configuration name
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, `optional`):
            An app authentication token with read access to all the datasets.
        committer_hf_token (`str`, `optional`):
            A user authentication token (See https://huggingface.co/settings/token) with write access. It must:
            - be part of the `huggingface` organization (to create the ref/convert/parquet "branch")
            - be part of the `datasets-maintainers` organization (to push to the ref/convert/parquet "branch")
        revision (`str`):
            The git revision (e.g. "main" or sha) of the dataset
        supported_datasets (`List[str]`):
            The list of supported datasets, saving the blocked datasets. If empty, all datasets are supported
            (saving the blocked datasets).
        blocked_datasets (`List[str]`):
            The list of blocked datasets. If empty, no dataset is blocked.
        max_dataset_size (`int`):
            The maximum size of a dataset in bytes. If the dataset is under the limit (which means that the size
            can be fetched), it will be allowed.
    Returns:
        `ParquetResponseResult`: An object with the parquet_response
          (dataset and list of parquet files) and the dataset_git_revision (sha) if any.
    <Tip>
    Raises the following errors:
        - [`~job_runners.config.parquet_and_info.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
        - [`~job_runners.config.parquet_and_info.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~job_runners.config.parquet_and_info.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
        - [`ValueError`]
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~job_runners.config.parquet_and_info.DatasetTooBigFromDatasetsError`]
            If the dataset is too big to be converted to parquet
    </Tip>
    """
    raise_if_blocked(dataset=dataset, blocked_datasets=blocked_datasets)
    ask_access(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=committer_hf_token)
    dataset_info = get_dataset_info_or_raise(
        dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, revision=revision
    )
    if dataset in supported_datasets:
        return
    raise_if_too_big_from_datasets(
        dataset=dataset,
        config=config,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        revision=revision,
        max_dataset_size=max_dataset_size,
    )
    raise_if_too_big_from_hub(dataset_info=dataset_info, max_dataset_size=max_dataset_size)


class EmptySplitsError(Exception):
    pass


class SplitInfoFormatError(Exception):
    pass


class EmptyConfigNameError(Exception):
    pass


class EmptyDownloadSizeError(Exception):
    pass


class EmptyFeaturesError(Exception):
    pass


class DatasetWithTooManyExternalFilesError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManyExternalFilesError", cause, True)


class DatasetWithTooBigExternalFilesError(ConfigParquetAndInfoJobRunnerError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooBigExternalFilesError", cause, True)


class UnsupportedExternalFilesError(ConfigParquetAndInfoJobRunnerError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "UnsupportedExternalFilesError", cause, True)


class ExternalFilesSizeRequestHTTPError(ConfigParquetAndInfoJobRunnerError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestHTTPError", cause, True)


class ExternalFilesSizeRequestConnectionError(ConfigParquetAndInfoJobRunnerError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestConnectionError", cause, True)


class ExternalFilesSizeRequestTimeoutError(ConfigParquetAndInfoJobRunnerError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestTimeoutError", cause, True)


class ExternalFilesSizeRequestError(ConfigParquetAndInfoJobRunnerError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestError", cause, True)


def _request_size(url: str, hf_token: Optional[str] = None) -> Optional[int]:
    headers = get_authentication_headers_for_url(url, use_auth_token=hf_token)
    response = http_head(url, headers=headers, max_retries=3)
    response.raise_for_status()
    size = response.headers.get("Content-Length") if response.ok else None
    return int(size) if size is not None else size


class _MockStreamingDownloadManager(StreamingDownloadManager):  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ext_data_files: List[str] = []

    def download(self, url_or_urls: Any) -> Any:
        url_or_urls = map_nested(
            self._download,
            url_or_urls,
            map_tuple=True,
            parallel_min_length=np.inf,
            # ^ parallel_min_length has int type, but is currently used in datasets for a comparison only
            # and it works with np.inf. No conversion is involved
            # (would raise: OverflowError: cannot convert float infinity to integer)
        )
        return url_or_urls

    def _download(self, urlpath: Any) -> str:
        urlpath_str = str(urlpath)
        if is_relative_path(urlpath_str):
            # append the relative path to the base_path
            urlpath_str = url_or_path_join(self._base_path, urlpath_str)
        elif not urlpath_str.startswith(self._base_path):
            # it's an external file
            self.ext_data_files.append(urlpath_str)
        return urlpath_str


def raise_if_too_big_from_external_data_files(
    builder: DatasetBuilder, max_dataset_size: int, max_external_data_files: int, hf_token: Optional[str]
) -> None:
    # Packaged dataset modules only load data files that are inside the dataset repository.
    # No need to check them since they're already caught by `raise_if_too_big_from_hub`
    if type(builder).__module__.startswith("datasets."):
        return
    # For datasets with a loading script however, we need to check the downloaded files
    mock_dl_manager = _MockStreamingDownloadManager(
        base_path=builder.base_path, download_config=DownloadConfig(use_auth_token=hf_token)
    )
    try:
        builder._split_generators(mock_dl_manager)
    except (requests.exceptions.RequestException, NotImplementedError) as error:
        if isinstance(error, NotImplementedError):
            # we can ignore the errors from functions not implemented in streaming mode like `.extract()` on TAR files
            if "is not implemented in streaming mode." not in str(error):
                raise UnsupportedExternalFilesError(
                    (
                        "Couldn't get the list of external files in `_split_generators` because it doesn't support"
                        f" streaming:\n{error}"
                    ),
                    error,
                ) from error
        elif isinstance(error, requests.exceptions.HTTPError):
            raise ExternalFilesSizeRequestHTTPError(
                (
                    "Couldn't get the list of external files in `_split_generators` because a request"
                    f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                    " (e.g. inside a data/ folder)."
                ),
                error,
            ) from error
        elif isinstance(error, requests.exceptions.ConnectionError):
            raise ExternalFilesSizeRequestConnectionError(
                (
                    "Couldn't get the list of external files in `_split_generators` because a request"
                    f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                    " (e.g. inside a data/ folder)."
                ),
                error,
            ) from error
        elif isinstance(error, requests.exceptions.Timeout):
            raise ExternalFilesSizeRequestTimeoutError(
                (
                    "Couldn't get the list of external files in `_split_generators` because a request"
                    f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                    " (e.g. inside a data/ folder)."
                ),
                error,
            ) from error
        else:
            raise ExternalFilesSizeRequestError(
                (
                    "Couldn't get the list of external files in `_split_generators` because a request"
                    f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                    " (e.g. inside a data/ folder)."
                ),
                error,
            ) from error
    ext_data_files = mock_dl_manager.ext_data_files
    if len(ext_data_files) > max_external_data_files:
        raise DatasetWithTooManyExternalFilesError(
            f"The conversion to parquet is limited to datasets with less than {max_external_data_files} files. "
            f"However it uses {len(ext_data_files)} data files."
        )
    elif ext_data_files:
        try:
            with ThreadPool(16) as pool:
                total_size = 0
                get_size = partial(_request_size, hf_token=hf_token)
                for i, size in enumerate(pool.imap_unordered(get_size, ext_data_files)):
                    if size is not None:
                        total_size += size
                        if total_size > max_dataset_size:
                            raise DatasetWithTooBigExternalFilesError(
                                f"The conversion to parquet is limited to datasets under {max_dataset_size} bytes."
                                f" However {i + 1} data files of {len(ext_data_files)} are already bigger than"
                                f" {total_size} bytes."
                            )
        except requests.exceptions.RequestException as error:
            if isinstance(error, requests.exceptions.HTTPError):
                raise ExternalFilesSizeRequestHTTPError(
                    (
                        "Couldn't get the size of external files in `_split_generators` because a request"
                        f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                        " (e.g. inside a data/ folder)."
                    ),
                    error,
                ) from error
            elif isinstance(error, requests.exceptions.ConnectionError):
                raise ExternalFilesSizeRequestConnectionError(
                    (
                        "Couldn't get the size of external files in `_split_generators` because a request"
                        f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                        " (e.g. inside a data/ folder)."
                    ),
                    error,
                ) from error
            elif isinstance(error, requests.exceptions.Timeout):
                raise ExternalFilesSizeRequestTimeoutError(
                    (
                        "Couldn't get the size of external files in `_split_generators` because a request"
                        f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                        " (e.g. inside a data/ folder)."
                    ),
                    error,
                ) from error
            else:
                raise ExternalFilesSizeRequestError(
                    (
                        "Couldn't get the size of external files in `_split_generators` because a request"
                        f" failed:\n{error}\nPlease consider moving your data files in this dataset repository instead"
                        " (e.g. inside a data/ folder)."
                    ),
                    error,
                ) from error


def compute_config_parquet_and_info_response(
    dataset: str,
    config: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    committer_hf_token: Optional[str],
    source_revision: str,
    target_revision: str,
    commit_message: str,
    url_template: str,
    supported_datasets: List[str],
    blocked_datasets: List[str],
    max_dataset_size: int,
    max_external_data_files: int,
) -> ConfigParquetAndInfoResponse:
    """
    Get the response of /parquet-and-dataset-info for one specific dataset on huggingface.co.
    It is assumed that the dataset can be accessed with the token.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            Dataset configuration name
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, `optional`):
            An app authentication token with read access to all the datasets.
        committer_hf_token (`str`, `optional`):
            A user authentication token (See https://huggingface.co/settings/token) with write access. It must:
            - be part of the `huggingface` organization (to create the ref/convert/parquet "branch")
            - be part of the `datasets-maintainers` organization (to push to the ref/convert/parquet "branch")
        source_revision (`str`):
            The git revision (e.g. "main" or sha) of the dataset used to prepare the parquet files
        target_revision (`str`):
            The target git revision (e.g. "ref/convert/parquet") of the dataset where to store the parquet files
        commit_message (`str`):
            The commit message to use when storing the parquet files
        url_template (`str`):
            The template to use to build the parquet file url
        supported_datasets (`List[str]`):
            The list of supported datasets, saving the blocked datasets. If empty, all datasets are supported
            (saving the blocked datasets).
        blocked_datasets (`List[str]`):
            The list of blocked datasets. If empty, no dataset is blocked.
        max_dataset_size (`int`):
            The maximum size of a dataset in bytes. If the dataset is under the limit (which means that the size
            can be fetched), it will be allowed.
        max_external_data_files (`int`):
            The maximum number of external data files of a dataset. This is for datasets with loading scripts only.
    Returns:
        `ConfigParquetAndInfoResponse`: An object with the config_parquet_and_info_response
          (dataset info and list of parquet files).
    <Tip>
    Raises the following errors:
        - [`~job_runners.config.parquet_and_info.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['HTTPError'](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError): any other error when
            asking access
        - [`~job_runners.config.parquet_and_info.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~job_runners.config.parquet_and_info.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~job_runners.config.parquet_and_info.DatasetTooBigFromDatasetsError`]
            If the dataset is too big to be converted to parquet
        - [`~job_runners.config.parquet_and_info.EmptyDatasetError`]
          The dataset is empty.
        - [`~job_runners.config.parquet_and_info.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
        - [`~job_runners.config.parquet_and_info.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`~job_runners.config.parquet_and_info.DatasetWithTooManyExternalFilesError`]
            If the dataset has too many external files to be converted to parquet
        - [`~job_runners.config.parquet_and_info.DatasetWithTooBigExternalFilesError`]
            If the dataset is too big external files be converted to parquet
        - [`~job_runners.config.parquet_and_info.UnsupportedExternalFilesError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`~job_runners.config.parquet_and_info.ExternalFilesSizeRequestHTTPError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`~job_runners.config.parquet_and_info.ExternalFilesSizeRequestConnectionError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`~job_runners.config.parquet_and_info.ExternalFilesSizeRequestTimeoutError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`~job_runners.config.parquet_and_info.ExternalFilesSizeRequestError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`~job_runners.config.parquet_and_info.PreviousStepStatusError`]
          If the previous step gave an error.
        - [`~job_runners.config.parquet_and_info.PreviousStepFormatError`]
            If the content of the previous step has not the expected format

    </Tip>
    """
    logging.info(f"get parquet files and dataset info for {dataset=} {config=}")

    raise_if_not_supported(
        dataset=dataset,
        config=config,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        committer_hf_token=committer_hf_token,
        revision=source_revision,
        supported_datasets=supported_datasets,
        blocked_datasets=blocked_datasets,
        max_dataset_size=max_dataset_size,
    )

    logging.info(f"get config names for {dataset=}")
    previous_step = "/config-names"
    try:
        response = get_response(kind=previous_step, dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError(f"No response found in previous step '{previous_step}' for this dataset.", e) from e
    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(f"Previous step {previous_step} gave an error: {response['http_status']}..")

    config_names_content = response["content"]
    if "config_names" not in config_names_content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    if not isinstance(config_names_content["config_names"], list):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"config_names should be a list, but got {type(config_names_content['config_names'])}"),
        )

    config_names = {config_name_item["config"] for config_name_item in config_names_content["config_names"]}
    if config not in config_names:
        raise ConfigNamesError(f"{config=} does not exist in {dataset=}")

    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)
    committer_hf_api = HfApi(endpoint=hf_endpoint, token=committer_hf_token)

    # prepare the parquet files locally
    local_parquet_files: List[ParquetFile] = []
    download_config = DownloadConfig(delete_extracted=True)
    try:
        builder = load_dataset_builder(
            path=dataset,
            name=config,
            revision=source_revision,
            use_auth_token=hf_token,
            download_config=download_config,
        )
    except _EmptyDatasetError as err:
        raise EmptyDatasetError(f"{dataset=} is empty.", cause=err) from err
    raise_if_too_big_from_external_data_files(
        builder=builder,
        max_dataset_size=max_dataset_size,
        max_external_data_files=max_external_data_files,
        hf_token=hf_token,
    )
    builder.download_and_prepare(file_format="parquet")  # the parquet files are stored in the cache dir
    dataset_info = asdict(builder.info)
    local_parquet_files.extend(
        ParquetFile(local_file=local_file, local_dir=builder.cache_dir, config=config)
        for local_file in glob.glob(f"{builder.cache_dir}**/*.parquet")
    )

    # create the target revision if it does not exist yet (clone from initial commit to avoid cloning all repo's files)
    try:
        refs = hf_api.list_repo_refs(repo_id=dataset, repo_type=DATASET_TYPE)
        if all(ref.ref != target_revision for ref in refs.converts):
            initial_commit = hf_api.list_repo_commits(repo_id=dataset, repo_type=DATASET_TYPE)[-1].commit_id
            committer_hf_api.create_branch(
                repo_id=dataset, branch=target_revision, repo_type=DATASET_TYPE, revision=initial_commit
            )
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err

    target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
    # - get repo parquet files
    all_repo_files: Set[str] = {f.rfilename for f in target_dataset_info.siblings}
    repo_parquet_files: Set[str] = {file for file in all_repo_files if file.endswith(".parquet")}
    # - get parquet files for current config
    config_files_to_add: Dict[str, str] = {
        parquet_file.repo_file(): parquet_file.local_file for parquet_file in local_parquet_files
    }
    # - get files that will be preserved in repo: files belonging to other configs and .gitattributes
    files_to_ignore: Set[str] = {
        file for config in config_names for file in repo_parquet_files if file.startswith(f"{config}/")
    }.union(".gitattributes")
    # - get files to be deleted - all files except for:
    #   - parquet files obtained for current config at this processing step,
    #   - parquet files belonging to other existing configs
    #   - .gitattributes
    files_to_delete = all_repo_files - set(config_files_to_add).union(files_to_ignore)
    delete_operations: List[CommitOperation] = [CommitOperationDelete(path_in_repo=file) for file in files_to_delete]
    logging.debug(f"{delete_operations=}")

    # send the files to the target revision
    add_operations: List[CommitOperation] = [
        CommitOperationAdd(path_in_repo=file, path_or_fileobj=local_file)
        for file, local_file in config_files_to_add.items()
    ]
    logging.debug(f"{add_operations=}")

    committer_hf_api.create_commit(
        repo_id=dataset,
        repo_type=DATASET_TYPE,
        revision=target_revision,
        operations=delete_operations + add_operations,
        commit_message=commit_message,
        parent_commit=target_dataset_info.sha,
    )

    # call the API again to get the list of parquet files
    target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=True)
    repo_files = [
        repo_file
        for repo_file in target_dataset_info.siblings
        if repo_file.rfilename.startswith(f"{config}/") and repo_file.rfilename.endswith(".parquet")
    ]
    # we might want to check if the sha of the parquet files is the same as the one we just uploaded
    # we could also check that the list of parquet files is exactly what we expect
    # let's not over engineer this for now. After all, what is on the Hub is the source of truth
    # and the /parquet response is more a helper to get the list of parquet files
    return ConfigParquetAndInfoResponse(
        parquet_files=[
            create_parquet_file_item(
                repo_file=repo_file,
                dataset=dataset,
                config=config,
                hf_endpoint=hf_endpoint,
                target_revision=target_revision,
                url_template=url_template,
            )
            for repo_file in repo_files
        ],
        dataset_info=dataset_info,
    )


class ConfigParquetAndInfoJobRunner(DatasetsBasedJobRunner):
    parquet_and_info_config: ParquetAndInfoConfig

    @staticmethod
    def get_job_type() -> str:
        return "config-parquet-and-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
        parquet_and_info_config: ParquetAndInfoConfig,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.parquet_and_info_config = parquet_and_info_config

    def compute(self) -> CompleteJobResult:
        if self.dataset is None:
            raise ParameterMissingError("'dataset' parameter is required")
        if self.config is None:
            raise ParameterMissingError("'config' parameter is required")
        return CompleteJobResult(
            compute_config_parquet_and_info_response(
                dataset=self.dataset,
                config=self.config,
                hf_endpoint=self.common_config.hf_endpoint,
                hf_token=self.common_config.hf_token,
                committer_hf_token=self.parquet_and_info_config.committer_hf_token,
                source_revision=self.parquet_and_info_config.source_revision,
                target_revision=self.parquet_and_info_config.target_revision,
                commit_message=self.parquet_and_info_config.commit_message,
                url_template=self.parquet_and_info_config.url_template,
                supported_datasets=self.parquet_and_info_config.supported_datasets,
                blocked_datasets=self.parquet_and_info_config.blocked_datasets,
                max_dataset_size=self.parquet_and_info_config.max_dataset_size,
                max_external_data_files=self.parquet_and_info_config.max_external_data_files,
            )
        )

    def get_new_splits(self, content: Mapping[str, Any]) -> Set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=self.dataset, config=self.config, split=split)
            for split in content["dataset_info"]["splits"]
        }
