# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import functools
import logging
import os
import re
from collections.abc import Callable, Generator
from contextlib import ExitStack
from fnmatch import fnmatch
from multiprocessing.pool import ThreadPool
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TypeVar, Union
from unittest.mock import patch
from urllib.parse import unquote

import datasets
import datasets.config
import datasets.info
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import DownloadConfig, Features, load_dataset_builder
from datasets.arrow_writer import ParquetWriter
from datasets.builder import DatasetBuilder, ManualDownloadError
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.download import StreamingDownloadManager
from datasets.packaged_modules.parquet.parquet import Parquet as ParquetBuilder
from datasets.splits import SplitDict, SplitInfo
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
    CommitOperationCopy,
    CommitOperationDelete,
)
from huggingface_hub.hf_api import CommitInfo, DatasetInfo, HfApi, RepoFile
from huggingface_hub.hf_file_system import HfFileSystem
from huggingface_hub.utils._errors import HfHubHTTPError, RepositoryNotFoundError
from libcommon.constants import (
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS,
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_BINARY_DATASETS,
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS,
    PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION,
)
from libcommon.dataset import get_dataset_info_for_supported_datasets
from libcommon.exceptions import (
    ConfigNamesError,
    CreateCommitError,
    DatasetInBlockListError,
    DatasetManualDownloadError,
    DatasetNotFoundError,
    DatasetWithTooManyParquetFilesError,
    EmptyDatasetError,
    ExternalFilesSizeRequestConnectionError,
    ExternalFilesSizeRequestError,
    ExternalFilesSizeRequestHTTPError,
    ExternalFilesSizeRequestTimeoutError,
    FileSystemError,
    LockedDatasetTimeoutError,
    PreviousStepFormatError,
    UnsupportedExternalFilesError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import lock
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import JobInfo, SplitHubFile
from tqdm.contrib.concurrent import thread_map

from worker.config import AppConfig, ParquetAndInfoConfig
from worker.dtos import CompleteJobResult, ConfigParquetAndInfoResponse
from worker.job_runners.config.config_job_runner import ConfigJobRunnerWithDatasetsCache
from worker.utils import (
    HF_HUB_HTTP_ERROR_RETRY_SLEEPS,
    LOCK_GIT_BRANCH_RETRY_SLEEPS,
    create_branch,
    hf_hub_url,
    retry,
)

DATASET_TYPE = "dataset"
MAX_FILES_PER_DIRECTORY = 10_000  # hf hub limitation
MAX_OPERATIONS_PER_COMMIT = 500

# For paths like "en/partial-train/0000.parquet" in the C4 dataset.
# Note that "-" is forbidden for split names so it doesn't create directory names collisions.
PARTIAL_SPLIT_PREFIX = "partial-"

T = TypeVar("T")


def repo_file_rfilename_sort_key(repo_file: RepoFile) -> str:
    if not isinstance(repo_file.rfilename, str):  # check type for mypy
        raise ValueError(f"Expected a string for repo_file.rfilename, but got a '{type(repo_file.rfilename)}'.")
    return repo_file.rfilename


class ParquetFile:
    def __init__(
        self, local_file: str, local_dir: str, config: str, split: str, shard_idx: int, partial: bool = False
    ):
        if not local_file.startswith(local_dir):
            raise ValueError(f"{local_file} is not in {local_dir}")
        if shard_idx >= MAX_FILES_PER_DIRECTORY:
            raise DatasetWithTooManyParquetFilesError(
                "The dataset has too many parquet files and can't be uploaded in the parquet directory "
                f"because it exceeds the maximum number of files per directory ({MAX_FILES_PER_DIRECTORY})."
            )
        self.local_file = local_file
        self.local_dir = local_dir
        self.config = config
        self.split = split
        self.shard_idx = shard_idx
        self.partial = partial

    @property
    def path_in_repo(self) -> str:
        partial_prefix = PARTIAL_SPLIT_PREFIX if self.partial else ""
        # Using 4 digits is ok since MAX_FILES_PER_DIRECTORY == 10_000
        return f"{self.config}/{partial_prefix}{self.split}/{self.shard_idx:04d}.parquet"


filename_pattern = re.compile("^[0-9]{4}\\.parquet$")


def parse_repo_filename(filename: str) -> tuple[str, str]:
    if not filename_pattern.match(os.path.basename(filename)):
        raise ValueError(f"Cannot parse {filename}")
    parts = filename.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid filename: {filename}")
    config, split, _ = parts
    if split.startswith(PARTIAL_SPLIT_PREFIX):
        split = split[len(PARTIAL_SPLIT_PREFIX) :]  # noqa: E203
    return config, split


def create_parquet_file_item(
    repo_file: RepoFile,
    dataset: str,
    config: str,
    hf_endpoint: str,
    target_revision: str,
    url_template: str,
) -> SplitHubFile:
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
    blocked_datasets: list[str],
) -> None:
    """
    Raise an error if the dataset is in the list of blocked datasets

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        blocked_datasets (`list[str]`):
            The list of blocked datasets. If empty, no dataset is blocked.
            Patterns are supported, e.g. "open-llm-leaderboard/*"

    Returns:
        `None`
    Raises the following errors:
        - [`libcommon.exceptions.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
    """
    for blocked_dataset in blocked_datasets:
        if fnmatch(dataset, blocked_dataset):
            raise DatasetInBlockListError(
                "The parquet conversion has been disabled for this dataset for now. Please open an issue in"
                " https://github.com/huggingface/datasets-server if you want this dataset to be supported."
            )


def is_parquet_builder_with_hub_files(builder: DatasetBuilder) -> bool:
    if not isinstance(builder, ParquetBuilder) or not builder.config.data_files:
        return False
    for split in builder.config.data_files:
        for data_file in builder.config.data_files[split]:
            if not data_file.startswith(f"hf://datasets/{builder.repo_id}@"):
                return False
    return True


def _is_too_big_from_hub(
    dataset_info: DatasetInfo,
    max_dataset_size: int,
) -> bool:
    """
    Raise an error if the dataset is too big to be converted to parquet, as measured by the sum of the repository
    files sizes given by the Hub.

    Args:
        dataset_info (`DatasetInfo`):
            The dataset info
        max_dataset_size (`int`):
            The maximum size of the dataset in bytes
    """
    dataset_size: int = sum(sibling.size for sibling in dataset_info.siblings if sibling.size is not None)
    return bool(dataset_size > max_dataset_size)


def _is_too_big_from_datasets(
    info: datasets.DatasetInfo,
    max_dataset_size: int,
) -> bool:
    """
    Raise an error if the dataset is too big to be converted to parquet, as measured by the sum of the configs
    sizes given by the datasets library

    Args:
        info (`datasets.DatasetInfo`):
            Dataset info from the datasets library
        max_dataset_size (`int`):
            The maximum size of the dataset in bytes
    """
    dataset_size = info.dataset_size if info.dataset_size is not None else 0
    return bool(dataset_size > max_dataset_size)


def raise_if_requires_manual_download(
    builder: DatasetBuilder,
    hf_endpoint: str,
    hf_token: Optional[str],
) -> None:
    """
    Raise an error if the dataset requires manual download.

    Args:
        builder (`datasets.builder.DatasetBuilder`):
            A dataset builder instance to check.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co").
        hf_token (`str`, *optional*):
            An app authentication token with read access to all the datasets.

    Returns:
        `None`

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError):
            If the datasets.config.HF_ENDPOINT is not set to the expected value.
        [`libcommon.exceptions.DatasetManualDownloadError`]:
            If the dataset requires manual download.
    """
    if datasets.config.HF_ENDPOINT != hf_endpoint:
        raise ValueError(
            f"Invalid datasets.config.HF_ENDPOINT value: '{datasets.config.HF_ENDPOINT}'. Please set it to:"
            f" '{hf_endpoint}'."
        )
    try:
        builder._check_manual_download(
            StreamingDownloadManager(base_path=builder.base_path, download_config=DownloadConfig(token=hf_token))
        )
    except ManualDownloadError as err:
        raise DatasetManualDownloadError(f"dataset={builder.repo_id} requires manual download.", cause=err) from err


def is_dataset_too_big(
    dataset_info: DatasetInfo,
    builder: DatasetBuilder,
    hf_endpoint: str,
    hf_token: Optional[str],
    max_dataset_size: int,
    max_external_data_files: int,
) -> bool:
    """
    Check:
    - the size of the dataset repository
    - the size in dataset info
    - the size and number of external files

    Args:
        dataset_info (`DatasetInfo`):
            The dataset info
        builder (`datasets.builder.DatasetBuilder`):
            A dataset builder instance to check.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, `optional`):
            An app authentication token with read access to all the datasets.
        revision (`str`):
            The git revision (e.g. "main" or sha) of the dataset
        max_dataset_size (`int`):
            The maximum size of a dataset in bytes. If the dataset is under the limit (which means that the size
            can be fetched), it will be allowed.
        max_external_data_files (`int`):
            The maximum number of external data files (i.e. not hosted on HF).
            If the dataset is under the limit (which means that the files can be fetched), it will be allowed.
    Returns:
        `ParquetResponseResult`: An object with the parquet_response
          (dataset and list of parquet files) and the dataset_git_revision (sha) if any.
    Raises the following errors:
        - [`libcommon.exceptions.UnsupportedExternalFilesError`]
          If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestHTTPError`]
          If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestConnectionError`]
          If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestTimeoutError`]
          If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestError`]
          If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          If the datasets.config.HF_ENDPOINT is not set to the expected value
    """
    if datasets.config.HF_ENDPOINT != hf_endpoint:
        raise ValueError(
            f"Invalid datasets.config.HF_ENDPOINT value: '{datasets.config.HF_ENDPOINT}'. Please set it to:"
            f" '{hf_endpoint}'."
        )
    return (
        _is_too_big_from_hub(dataset_info=dataset_info, max_dataset_size=max_dataset_size)
        or _is_too_big_from_datasets(
            builder.info,
            max_dataset_size=max_dataset_size,
        )
        or _is_too_big_from_external_data_files(
            builder=builder,
            max_dataset_size=max_dataset_size,
            max_external_data_files=max_external_data_files,
            hf_token=hf_token,
        )
    )


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


def _request_size(url: str, hf_token: Optional[str] = None) -> Optional[int]:
    headers = get_authentication_headers_for_url(url, token=hf_token)
    response = http_head(url, headers=headers, max_retries=3)
    response.raise_for_status()
    size = response.headers.get("Content-Length") if response.ok else None
    return int(size) if size is not None else size


class _MockStreamingDownloadManager(StreamingDownloadManager):  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ext_data_files: list[str] = []

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


def _is_too_big_from_external_data_files(
    builder: DatasetBuilder, max_dataset_size: int, max_external_data_files: int, hf_token: Optional[str]
) -> bool:
    # Packaged dataset modules only load data files that are inside the dataset repository.
    # No need to check them since they're already caught by `raise_if_too_big_from_hub`
    if type(builder).__module__.startswith("datasets."):
        return False
    # For datasets with a loading script however, we need to check the downloaded files
    mock_dl_manager = _MockStreamingDownloadManager(
        base_path=builder.base_path, download_config=DownloadConfig(token=hf_token)
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
        return True
    elif ext_data_files:
        try:
            with ThreadPool(16) as pool:
                total_size = 0
                get_size = functools.partial(_request_size, hf_token=hf_token)
                for i, size in enumerate(pool.imap_unordered(get_size, ext_data_files)):
                    if size is not None:
                        total_size += size
                        return total_size > max_dataset_size
                return False
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
    return False


def get_writer_batch_size_from_info(ds_config_info: datasets.info.DatasetInfo) -> Optional[int]:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files.
    The default in `datasets` is 1,000 but we lower it to 100 for image datasets.
    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.
    Args:
        ds_config_info (`datasets.info.DatasetInfo`):
            Dataset info from `datasets`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a dataset builder.
            If `None`, then it will use the `datasets` default.
    """
    if "Audio(" in str(ds_config_info.features):
        return PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS
    elif "Image(" in str(ds_config_info.features):
        return PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS
    elif "'binary'" in str(ds_config_info.features):
        return PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_ROW_GROUP_SIZE_FOR_BINARY_DATASETS
    else:
        return None


def get_writer_batch_size_from_row_group_size(
    num_rows: int, row_group_byte_size: int, max_row_group_byte_size: int, factor_of: int = 100, divide_step: int = 10
) -> int:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files,
    given a sample row group size that mught be too big.

    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.

    Args:
        num_rows (`int`):
            Number of rows in the sample row group.
        row_group_byte_size (`int`):
            Number of bytes of uncompressed data in the sample row group.
        max_row_group_byte_size (`int`):
            Maximum number of bytes of uncompressed data for batches that
            will be passed to a dataset builder.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a dataset builder.
    """
    writer_batch_size = max(num_rows // factor_of * factor_of, factor_of)
    writer_batch_byte_size = row_group_byte_size * writer_batch_size / num_rows
    while writer_batch_size > factor_of and writer_batch_byte_size > max_row_group_byte_size:
        writer_batch_size = max(writer_batch_size // divide_step // factor_of * factor_of, factor_of)
        writer_batch_byte_size = row_group_byte_size * writer_batch_size / num_rows
    return writer_batch_size


def copy_parquet_files(builder: DatasetBuilder) -> list[CommitOperationCopy]:
    """Copy parquet files by copying the git LFS pointer files"""
    data_files = builder.config.data_files
    if not data_files:
        raise EmptyDatasetError("Empty parquet data_files")
    parquet_operations = []
    total_num_parquet_files = sum(len(data_files[split]) for split in data_files)
    if total_num_parquet_files >= MAX_FILES_PER_DIRECTORY:
        raise DatasetWithTooManyParquetFilesError(
            f"The dataset has {total_num_parquet_files} parquet files and can't be linked in the parquet directory "
            f"because it exceeds the maximum number of files per directory ({MAX_FILES_PER_DIRECTORY})."
        )
    for split in data_files:
        for shard_idx, data_file in enumerate(data_files[split]):
            # data_file format for hub files is hf://datasets/{repo_id}@{revision}/{path_in_repo}
            src_revision, src_path_in_repo = data_file.split("@")[1].split("/", 1)
            src_revision = unquote(src_revision)
            src_path_in_repo = unquote(src_path_in_repo)
            path_in_repo = f"{builder.config.name}/{split}/{shard_idx:04d}.parquet"
            parquet_operations.append(
                CommitOperationCopy(
                    src_path_in_repo=src_path_in_repo, path_in_repo=path_in_repo, src_revision=src_revision
                )
            )
    return parquet_operations


class NotAParquetFileError(ValueError):
    """When a remote parquet file can't be parsed"""

    pass


class ParquetValidationError(ValueError):
    """When a parquet file is not validated for copy"""


class TooBigRowGroupsError(ParquetValidationError):
    """When a parquet file has row groups that are too big for copy"""

    def __init__(self, *args: object, num_rows: int, row_group_byte_size: int) -> None:
        super().__init__(*args)
        self.num_rows = num_rows
        self.row_group_byte_size = row_group_byte_size


def get_parquet_file_and_size(url: str, hf_endpoint: str, hf_token: Optional[str]) -> tuple[pq.ParquetFile, int]:
    fs = HfFileSystem(endpoint=hf_endpoint, token=hf_token)
    f = fs.open(url)
    return pq.ParquetFile(f), f.size


def retry_and_validate_get_parquet_file_and_size(
    url: str, hf_endpoint: str, hf_token: Optional[str], validate: Optional[Callable[[pq.ParquetFile], None]]
) -> tuple[pq.ParquetFile, int]:
    try:
        sleeps = [1, 1, 1, 10, 10, 10]
        pf, size = retry(on=[pa.ArrowInvalid], sleeps=sleeps)(get_parquet_file_and_size)(url, hf_endpoint, hf_token)
        if validate:
            validate(pf)
        return pf, size
    except RuntimeError as err:
        if err.__cause__ and isinstance(err.__cause__, pa.ArrowInvalid):
            raise NotAParquetFileError(f"Not a parquet file: '{url}'") from err.__cause__
        else:
            raise err


class ParquetFileValidator:
    """
    Validate the Parquet files before they are copied to the target revision.
    In particular we check that the row group size is not too big, otherwise the dataset viewer
    doesn't work correctly.

    Note: we only validate the first parquet files (default 5 first files).
    We don't want to check the biggest row group of all the dataset, but rather just get the order
    of magnitude of the size. Otherwise we might end up converting a dataset that has 99% good row
    groups but 1% that is a bit too big, which is overkill.
    """

    def __init__(self, max_row_group_byte_size: int, max_validation: int = 5) -> None:
        self.max_row_group_byte_size = max_row_group_byte_size
        self.num_validations = 0
        self.max_validations = max_validation

    def validate(self, pf: pq.ParquetFile) -> None:
        if self.num_validations >= self.max_validations:
            return
        row_group_metadata = pf.metadata.row_group(0)
        row_group_size = row_group_metadata.total_byte_size
        if row_group_metadata.total_byte_size > self.max_row_group_byte_size:
            raise TooBigRowGroupsError(
                (
                    f"Parquet file has too big row groups. First row group has {row_group_size} which exceeds the"
                    f" limit of {self.max_row_group_byte_size}"
                ),
                num_rows=row_group_metadata.num_rows,
                row_group_byte_size=row_group_metadata.total_byte_size,
            )
        self.num_validations += 1


def fill_builder_info(
    builder: DatasetBuilder,
    hf_endpoint: str,
    hf_token: Optional[str],
    validate: Optional[Callable[[pq.ParquetFile], None]],
) -> None:
    """Fill the builder DatasetInfo from the copied parquet files"""
    data_files = builder.config.data_files
    if not data_files:
        raise EmptyDatasetError("Empty parquet data_files")
    builder.info.builder_name = builder.name
    builder.info.dataset_name = builder.dataset_name
    builder.info.config_name = builder.config.name
    builder.info.version = builder.config.version
    builder.info.splits = SplitDict()
    builder.info.download_size = 0
    builder.info.dataset_size = 0
    for split in data_files:
        split = str(split)  # in case it's a NamedSplit
        try:
            parquet_files_and_sizes: list[tuple[pq.ParquetFile, int]] = thread_map(
                functools.partial(
                    retry_and_validate_get_parquet_file_and_size,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    validate=validate,
                ),
                data_files[split],
                unit="pq",
                disable=True,
            )
            parquet_files, sizes = zip(*parquet_files_and_sizes)
        except ParquetValidationError:
            raise
        except Exception as e:
            raise FileSystemError(f"Could not read the parquet files: {e}") from e
        if parquet_files:
            first_pf = parquet_files[0]
            if builder.info.features is None:
                builder.info.features = Features.from_arrow_schema(first_pf.schema_arrow)
            first_row_group = first_pf.read_row_group(0)
            compression_ratio = first_row_group.nbytes / first_row_group.num_rows
            num_examples = sum(parquet_file.metadata.num_rows for parquet_file in parquet_files)
            approx_num_bytes = int(compression_ratio * num_examples)
            builder.info.splits.add(SplitInfo(split, num_bytes=approx_num_bytes, num_examples=num_examples))
            builder.info.download_size += sum(sizes)
            builder.info.dataset_size += approx_num_bytes


class limit_parquet_writes:
    """
    Context manager that limits the number of bytes a `DatasetBuilder` can write to parquet.

    It works by monitoring the calls to `pq.ParquetWriter.write_table` and stopping
    the `GeneratorBasedBuilder._generate_examples` and `ArrowBasedBuilder._generate_tables`
    generators once we reach the maximum number of bytes.

    Since the generator is stopped after we reach the maximum number of bytes, the actual
    number of bytes generated might be slightly higher than the requested limit.

    Example of usage:

    ```python
    builder = load_dataset_builder("squad")
    max_dataset_size = 10_000_000
    with limit_parquet_writes(builder, max_dataset_size=max_dataset_size) as limiter:
        builder.download_and_prepare(file_format="parquet")
        assert builder.info.dataset_size == limiter.total_bytes < max_dataset_size + epsilon
    ```

    The limiter is usually used with a `StreamingDownloadManager` to not have to download
    the full dataset:

    ```python
    builder = load_dataset_builder("squad")
    max_dataset_size = 10_000_000
    dl_manager = StreamingDownloadManager(...)
    for split_generator in builder._split_generators(dl_manager):
        with limit_parquet_writes(builder, max_dataset_size=max_dataset_size):
            builder._prepare_split(split_generator=split_generator, file_format="parquet")
    ```
    """

    def __init__(
        self,
        builder: Union[datasets.builder.GeneratorBasedBuilder, datasets.builder.ArrowBasedBuilder],
        max_dataset_size: int,
    ) -> None:
        self.total_bytes = 0
        self.builder = builder
        self.max_dataset_size = max_dataset_size
        self.exit_stack = ExitStack()

    def __enter__(self) -> "limit_parquet_writes":
        limiter = self

        class _TrackedParquetWriter(pq.ParquetWriter):  # type: ignore
            """Count on-the-fly how many bytes are written"""

            def track_write_table(self, pa_table: pa.Table) -> None:
                limiter.total_bytes += pa_table.nbytes

            def write_table(self, pa_table: pa.Table, row_group_size: Optional[int] = None) -> None:
                self.track_write_table(pa_table)
                super().write_table(pa_table, row_group_size=row_group_size)

        def limited_generator(
            generator: Callable[..., Generator[T, None, None]]
        ) -> Callable[..., Generator[T, None, None]]:
            """Stop the underlying generator once we reach the maximum dataset size"""

            @functools.wraps(generator)
            def wrapped(*args: Any, **kwargs: Any) -> Generator[T, None, None]:
                for item in generator(*args, **kwargs):
                    if limiter.total_bytes < limiter.max_dataset_size:
                        yield item
                    else:
                        break

            return wrapped

        self.exit_stack.enter_context(patch.object(ParquetWriter, "_WRITER_CLASS", _TrackedParquetWriter))
        if isinstance(self.builder, datasets.builder.GeneratorBasedBuilder):
            self.exit_stack.enter_context(
                patch.object(self.builder, "_generate_examples", limited_generator(self.builder._generate_examples))
            )
        else:
            self.exit_stack.enter_context(
                patch.object(self.builder, "_generate_tables", limited_generator(self.builder._generate_tables))
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        return self.exit_stack.close()


def list_generated_parquet_files(builder: DatasetBuilder, partial: bool = False) -> list[ParquetFile]:
    """List the parquet files generated by `builder.download_and_prepare` in the `builder.cache_dir`."""
    if not builder.info.splits:
        raise EmptyDatasetError("No split found after generating parquet files")
    split_dict = builder.info.splits
    local_parquet_files: list[ParquetFile] = []
    for split, split_info in split_dict.items():
        # We know the `datasets` library uses a template for the shards names:
        # - {builder.dataset_name}-{split}.parquet if there is only one shard
        # - {builder.dataset_name}-{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet otherwise
        num_shards = len(split_info.shard_lengths) if isinstance(split_info.shard_lengths, list) else 1
        filename_suffix = "-{shard_idx:05d}-of-" + f"{num_shards:05d}" if num_shards > 1 else ""
        filename = f"{builder.dataset_name}-{split}{filename_suffix}.parquet"
        local_parquet_files.extend(
            [
                ParquetFile(
                    local_file=os.path.join(
                        builder.cache_dir,
                        filename.format(shard_idx=shard_idx),
                    ),
                    local_dir=builder.cache_dir,
                    config=builder.config.name,
                    split=split,
                    shard_idx=shard_idx,
                    partial=partial,
                )
                for shard_idx in range(num_shards)
            ]
        )
    return local_parquet_files


def stream_convert_to_parquet(
    builder: DatasetBuilder, max_dataset_size: Optional[int], writer_batch_size: Optional[int] = None
) -> tuple[list[CommitOperationAdd], bool]:
    """Stream and prepare the dataset as parquet files and fills the builder info."""
    writer_batch_size = writer_batch_size or get_writer_batch_size_from_info(builder.info)
    if writer_batch_size is not None and (
        builder._writer_batch_size is None or builder._writer_batch_size > writer_batch_size
    ):
        builder._writer_batch_size = writer_batch_size
    dl_manager = StreamingDownloadManager(
        base_path=builder.base_path,
        download_config=DownloadConfig(token=builder.token, storage_options=builder.storage_options),
        dataset_name=builder.name,
        data_dir=builder.config.data_dir,
    )
    os.makedirs(builder.cache_dir, exist_ok=True)
    split_dict = SplitDict(dataset_name=builder.name)
    splits_generators = {sg.name: sg for sg in builder._split_generators(dl_manager)}
    prepare_split_kwargs: dict[str, Any] = (
        {"check_duplicate_keys": True} if isinstance(builder, datasets.builder.GeneratorBasedBuilder) else {}
    )
    partial = False
    for split in splits_generators:
        split_dict.add(splits_generators[split].split_info)
        if max_dataset_size is None:
            builder._prepare_split(
                split_generator=splits_generators[split], file_format="parquet", **prepare_split_kwargs
            )
        else:
            with limit_parquet_writes(builder, max_dataset_size=max_dataset_size) as limiter:
                builder._prepare_split(
                    split_generator=splits_generators[split], file_format="parquet", **prepare_split_kwargs
                )
                partial = partial or limiter.total_bytes >= max_dataset_size
    builder.info.splits = split_dict
    builder.info.dataset_size = sum(split.num_bytes for split in builder.info.splits.values())
    builder.info.download_size = None
    builder.info.size_in_bytes = None

    # send the files to the target revision
    local_parquet_files = list_generated_parquet_files(builder, partial=partial)
    parquet_operations: list[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo=parquet_file.path_in_repo, path_or_fileobj=parquet_file.local_file)
        for parquet_file in local_parquet_files
    ]
    return parquet_operations, partial


def convert_to_parquet(builder: DatasetBuilder) -> list[CommitOperationAdd]:
    """Download and prepare the dataset as parquet files and fills the builder info."""
    # prepare the parquet files locally
    writer_batch_size = get_writer_batch_size_from_info(builder.info)
    if writer_batch_size is not None and (
        builder._writer_batch_size is None or builder._writer_batch_size > writer_batch_size
    ):
        builder._writer_batch_size = writer_batch_size
    builder.download_and_prepare(
        file_format="parquet"
    )  # the parquet files are stored in the cache dir and it fills the info
    local_parquet_files = list_generated_parquet_files(builder)

    # send the files to the target revision
    parquet_operations: list[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo=parquet_file.path_in_repo, path_or_fileobj=parquet_file.local_file)
        for parquet_file in local_parquet_files
    ]
    logging.debug(f"{parquet_operations=}")
    return parquet_operations


def create_commits(
    hf_api: HfApi,
    repo_id: str,
    operations: list[CommitOperation],
    *,
    commit_message: str,
    revision: Optional[str] = None,
    parent_commit: Optional[str] = None,
    max_operations_per_commit: int = MAX_OPERATIONS_PER_COMMIT,
) -> list[CommitInfo]:
    """
    Creates one or several commits in the given dataset repo, deleting & uploading files as needed.

    Args:
        hf_api (`huggingface_hub.HfApi`):
            The HfApi to use to commit the operations.
        repo_id (`str`):
            The repository in which the commit will be created, for example:
            `"username/my_dataset"`
        operations (`Iterable` of [`huggingface_hub.hf_api.CommitOperation`]):
            An iterable of operations to include in the commit, either:

                - [`huggingface_hub.hf_api.CommitOperationAdd`] to upload a file
                - [`huggingface_hub.hf_api.CommitOperationDelete`] to delete a file
                - [`huggingface_hub.hf_api.CommitOperationCopy`] to copy a file
        commit_message (`str`):
            The summary (first line) of the commit that will be created.
        commit_description (`str`, *optional*):
            The description of the commit that will be created
        token (`str`, *optional*):
            Authentication token, obtained with `HfApi.login` method. Will
            default to the stored token.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or
            space, `None` or `"model"` if uploading to a model. Default is
            `None`.
        revision (`str`, *optional*):
            The git revision to commit from. Defaults to the head of the `"main"` branch.
        parent_commit (`str`, *optional*):
            The OID / SHA of the parent commit, as a hexadecimal string.
            Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`,
            the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr`
            is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit`
            ensures the repo has not changed before committing the changes, and can be especially useful
            if the repo is updated / committed to concurrently.
        max_operations_per_commit (`int`, *optional*):
            The max number of operations per commit, to avoid time out errors from the Hub. Defaults to 500.
    Returns:
        [`list[huggingface_hub.CommitInfo]`]:
            List of [`CommitInfo`] containing information about the newly created commit (commit hash, commit
            url, pr url, commit message,...).
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If commit message is empty.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If parent commit is not a valid commit OID.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API returns an HTTP 400 error (bad request)
        [`huggingface_hub.utils.RepositoryNotFoundError`]:
            If repository is not found (error 404): wrong repo_id/repo_type, private
            but not authenticated or repo does not exist.
        [`libcommon.exceptions.CreateCommitError`]:
            If one of the commits could not be created on the Hub.
    """
    commit_infos: list[CommitInfo] = []
    offsets = range(0, len(operations), max_operations_per_commit)
    for commit_idx, offset in enumerate(offsets):
        batch_msg = f" (step {commit_idx + 1} of {len(offsets)})" if len(offsets) > 1 else ""
        retry_create_commit = retry(on=[HfHubHTTPError], sleeps=HF_HUB_HTTP_ERROR_RETRY_SLEEPS)(hf_api.create_commit)
        try:
            commit_info = retry_create_commit(
                repo_id=repo_id,
                repo_type=DATASET_TYPE,
                revision=revision,
                operations=operations[offset : offset + max_operations_per_commit],  # noqa: E203
                commit_message=commit_message + batch_msg,
                parent_commit=commit_infos[-1].oid if commit_infos else parent_commit,
            )
        except RuntimeError as e:
            if e.__cause__ and isinstance(e.__cause__, HfHubHTTPError):
                raise CreateCommitError(
                    message=(
                        f"Commit {commit_idx}/{len(offsets)} could not be created on the Hub (after"
                        f" {len(HF_HUB_HTTP_ERROR_RETRY_SLEEPS)} attempts)."
                    ),
                    cause=e.__cause__,
                ) from e.__cause__
            raise e
        commit_infos.append(commit_info)
    return commit_infos


def get_delete_operations(
    parquet_operations: list[CommitOperationAdd], all_repo_files: set[str], config_names: set[str], config: str
) -> list[CommitOperationDelete]:
    # - get files that will be preserved in repo:
    #   1. parquet files belonging to any other config (otherwise outdated files might be preserved)
    #   2. duckdb files belonging to any config
    #   3. .gitattributes
    pattern_in_any_config_dir = re.compile(f"^({'|'.join(re.escape(conf) for conf in config_names)})/")
    pattern_in_any_other_config_dir = re.compile(
        f"^({'|'.join(re.escape(conf) for conf in config_names.difference({config}))})/"
    )
    files_to_ignore: set[str] = {
        file
        for file in all_repo_files
        if (pattern_in_any_other_config_dir.match(file) and file.endswith(".parquet"))
        or (pattern_in_any_config_dir.match(file) and file.endswith(".duckdb"))
    }.union({".gitattributes"})
    # - get files to be deleted - all files except for:
    #   - the files to be preserved
    #   - parquet files obtained for current config at this processing step
    files_to_add = [operation.path_in_repo for operation in parquet_operations]
    files_to_delete = all_repo_files - set(files_to_add).union(files_to_ignore)
    delete_operations = [CommitOperationDelete(path_in_repo=file) for file in files_to_delete]
    logging.debug(f"{delete_operations=}")
    return delete_operations


def commit_parquet_conversion(
    hf_api: HfApi,
    committer_hf_api: HfApi,
    dataset: str,
    config: str,
    config_names: set[str],
    parquet_operations: list[CommitOperation],
    commit_message: str,
    target_revision: Optional[str],
) -> list[CommitInfo]:
    """
    Creates one or several commits in the given dataset repo, deleting & uploading files as needed.

    Args:
        hf_api (`huggingface_hub.HfApi`):
            The HfApi to get the dataset info.
        committer_hf_api (`huggingface_hub.HfApi`):
            The HfApi to use to commit the operations.
        dataset (`str`):
            The dataset in which the commit will be created, for example:
            `"username/my_dataset"`
        config (`str`):
            The dataset configuration.
        config_names (`list[str]`):
            The list of all the configurations of this dataset. This is used to clean
            the other fiels and directories in the repo, if any.
        parquet_operations (`list[huggingface_hub.hf_api.CommitOperation]`):
            List of commit operation for the parquet conversion. It could be
            file additions or file copies for example.
        commit_message (`str`):
            The summary (first line) of the commit that will be created.
        target_revision (`str`, *optional*):
            The git revision to commit from. Defaults to the head of the `"main"` branch.
    Returns:
        [`list[huggingface_hub.CommitInfo]`]:
            List of [`CommitInfo`] containing information about the newly created commit (commit hash, commit
            url, pr url, commit message,...).
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If commit message is empty.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If parent commit is not a valid commit OID.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API returns an HTTP 400 error (bad request)
        [`huggingface_hub.utils.RepositoryNotFoundError`]:
            If repository is not found (error 404): wrong repo_id/repo_type, private
            but not authenticated or repo does not exist.
        [`libcommon.exceptions.CreateCommitError`]:
            If one of the commits could not be created on the Hub.
    """
    target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
    all_repo_files: set[str] = {f.rfilename for f in target_dataset_info.siblings}
    delete_operations = get_delete_operations(
        parquet_operations=parquet_operations, all_repo_files=all_repo_files, config_names=config_names, config=config
    )
    operations = delete_operations + parquet_operations
    return create_commits(
        committer_hf_api,
        repo_id=dataset,
        revision=target_revision,
        operations=operations,
        commit_message=commit_message,
        parent_commit=target_dataset_info.sha,
    )


def compute_config_parquet_and_info_response(
    job_id: str,
    dataset: str,
    config: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    committer_hf_token: Optional[str],
    source_revision: str,
    target_revision: str,
    commit_message: str,
    url_template: str,
    blocked_datasets: list[str],
    max_dataset_size: int,
    max_external_data_files: int,
    max_row_group_byte_size_for_copy: int,
    no_max_size_limit_datasets: list[str],
) -> ConfigParquetAndInfoResponse:
    """
    Get the response of config-parquet-and-info for one specific dataset and config on huggingface.co.
    It is assumed that the dataset can be accessed with the token.
    Args:
        job_id (`str`):
            The id of the current Job. It is used to lock the access to the parquet conversion branch on the Hub.
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
            An app authentication token with write access. It must be part of the `datasets-maintainers`
              organization (to create the ref/convert/parquet "branch" and push to it)
        source_revision (`str`):
            The git revision (e.g. "main" or sha) of the dataset used to prepare the parquet files
        target_revision (`str`):
            The target git revision (e.g. "ref/convert/parquet") of the dataset where to store the parquet files
        commit_message (`str`):
            The commit message to use when storing the parquet files
        url_template (`str`):
            The template to use to build the parquet file url
        blocked_datasets (`list[str]`):
            The list of blocked datasets. If empty, no dataset is blocked.
        max_dataset_size (`int`):
            The maximum size of a dataset in bytes. If the dataset is under the limit (which means that the size
            can be fetched), it will be allowed.
        max_external_data_files (`int`):
            The maximum number of external data files of a dataset. This is for datasets with loading scripts only.
        max_row_group_byte_size_for_copy (`int`):
            The maximum size in bytes of parquet files that are allowed to be copied without being converted.
        no_max_size_limit_datasets (`list[str]`):
            List of datasets that should be fully converted (no partial conversion).
    Returns:
        `ConfigParquetAndInfoResponse`: An object with the config_parquet_and_info_response
          (dataset info and list of parquet files).
    Raises the following errors:
        - [`libcommon.exceptions.DatasetNotFoundError`]:
          if the dataset does not exist, or if the token does not give the sufficient access to the dataset,
        - ['requests.exceptions.HTTPError'](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
          any other error when asking access
        - [`libcommon.simple_cache.CachedArtifactError`]
            If the previous step gave an error.
        - [`libcommon.exceptions.CreateCommitError`]:
          If one of the commits could not be created on the Hub.
        - [`libcommon.exceptions.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`libcommon.exceptions.DatasetManualDownloadError`]:
          If the dataset requires manual download.
        - [`libcommon.exceptions.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`libcommon.exceptions.DatasetTooBigFromDatasetsError`]
          If the dataset is too big to be converted to parquet, as measured by the sum of the configs
          sizes given by the datasets library.
        - [`libcommon.exceptions.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet, as measured by the sum of the repository
          files sizes given by the Hub.
        - [`libcommon.exceptions.EmptyDatasetError`]
          The dataset is empty.
        - [`libcommon.exceptions.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
        - [`libcommon.exceptions.DatasetWithTooManyExternalFilesError`]
            If the dataset has too many external files to be converted to parquet
        - [`libcommon.exceptions.DatasetWithTooBigExternalFilesError`]
            If the dataset is too big external files be converted to parquet
        - [`libcommon.exceptions.UnsupportedExternalFilesError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestHTTPError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestConnectionError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestTimeoutError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.ExternalFilesSizeRequestError`]
            If we failed to get the external files sizes to make sure we can convert the dataset to parquet
        - [`libcommon.exceptions.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the datasets.config.HF_ENDPOINT is not set to the expected value
    """
    logging.info(f"get parquet files and dataset info for {dataset=} {config=}")

    raise_if_blocked(dataset=dataset, blocked_datasets=blocked_datasets)

    logging.info(f"getting config names for {dataset=}")
    previous_step = "dataset-config-names"
    config_names_best_response = get_previous_step_or_raise(kinds=[previous_step], dataset=dataset)

    config_names_content = config_names_best_response.response["content"]
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

    download_config = DownloadConfig(delete_extracted=True)
    try:
        builder = load_dataset_builder(
            path=dataset,
            name=config,
            revision=source_revision,
            token=hf_token,
            download_config=download_config,
        )
    except _EmptyDatasetError as err:
        raise EmptyDatasetError(f"{dataset=} is empty.", cause=err) from err
    except FileNotFoundError as err:
        raise DatasetNotFoundError("The dataset, or the revision, does not exist on the Hub.") from err

    partial = False
    if is_parquet_builder_with_hub_files(builder):
        try:
            parquet_operations = copy_parquet_files(builder)
            validate = ParquetFileValidator(max_row_group_byte_size=max_row_group_byte_size_for_copy).validate
            fill_builder_info(builder, hf_endpoint=hf_endpoint, hf_token=hf_token, validate=validate)
        except TooBigRowGroupsError as err:
            # aim for a writer_batch_size that is factor of 100
            # and with a batch_byte_size that is smaller than max_row_group_byte_size_for_copy
            writer_batch_size = get_writer_batch_size_from_row_group_size(
                num_rows=err.num_rows,
                row_group_byte_size=err.row_group_byte_size,
                max_row_group_byte_size=max_row_group_byte_size_for_copy,
            )
            parquet_operations, partial = stream_convert_to_parquet(
                builder,
                max_dataset_size=None if dataset in no_max_size_limit_datasets else max_dataset_size,
                writer_batch_size=writer_batch_size,
            )
    else:
        raise_if_requires_manual_download(
            builder=builder,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
        )
        dataset_info = get_dataset_info_for_supported_datasets(
            dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, revision=source_revision, files_metadata=True
        )
        if is_dataset_too_big(
            dataset_info=dataset_info,
            builder=builder,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            max_dataset_size=max_dataset_size,
            max_external_data_files=max_external_data_files,
        ):
            parquet_operations, partial = stream_convert_to_parquet(
                builder, max_dataset_size=None if dataset in no_max_size_limit_datasets else max_dataset_size
            )
        else:
            parquet_operations = convert_to_parquet(builder)

    try:
        # ^ timeouts after ~7 minutes
        with lock.git_branch(
            dataset=dataset, branch=target_revision, owner=job_id, sleeps=LOCK_GIT_BRANCH_RETRY_SLEEPS
        ):
            # create the target revision if we managed to get the parquet files and it does not exist yet
            # (clone from initial commit to avoid cloning all repo's files)
            create_branch(
                dataset=dataset,
                target_revision=target_revision,
                hf_api=hf_api,
                committer_hf_api=committer_hf_api,
            )

            # commit the parquet files
            commit_parquet_conversion(
                hf_api=hf_api,
                committer_hf_api=committer_hf_api,
                dataset=dataset,
                config=config,
                parquet_operations=parquet_operations,
                config_names=config_names,
                target_revision=target_revision,
                commit_message=commit_message,
            )
            # call the API again to get the list of parquet files
            target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=True)
    except TimeoutError as err:
        raise LockedDatasetTimeoutError("the dataset is currently locked, please try again later.") from err
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub (was deleted during job).") from err

    repo_files = [
        repo_file
        for repo_file in target_dataset_info.siblings
        if repo_file.rfilename.startswith(f"{config}/") and repo_file.rfilename.endswith(".parquet")
    ]
    repo_files.sort(key=repo_file_rfilename_sort_key)
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
        dataset_info=asdict(builder.info),
        partial=partial,
    )


class ConfigParquetAndInfoJobRunner(ConfigJobRunnerWithDatasetsCache):
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
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.parquet_and_info_config = app_config.parquet_and_info

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_config_parquet_and_info_response(
                job_id=self.job_info["job_id"],
                dataset=self.dataset,
                config=self.config,
                hf_endpoint=self.app_config.common.hf_endpoint,
                hf_token=self.app_config.common.hf_token,
                committer_hf_token=self.parquet_and_info_config.committer_hf_token,
                source_revision=self.parquet_and_info_config.source_revision,
                target_revision=self.parquet_and_info_config.target_revision,
                commit_message=self.parquet_and_info_config.commit_message,
                url_template=self.parquet_and_info_config.url_template,
                blocked_datasets=self.parquet_and_info_config.blocked_datasets,
                max_dataset_size=self.parquet_and_info_config.max_dataset_size,
                max_external_data_files=self.parquet_and_info_config.max_external_data_files,
                max_row_group_byte_size_for_copy=self.parquet_and_info_config.max_row_group_byte_size_for_copy,
                no_max_size_limit_datasets=self.parquet_and_info_config.no_max_size_limit_datasets,
            )
        )
