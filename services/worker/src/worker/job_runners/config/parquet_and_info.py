# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import glob
import logging
import re
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import datasets
import datasets.config
import datasets.info
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import DownloadConfig, Features, load_dataset_builder
from datasets.builder import DatasetBuilder, ManualDownloadError
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.data_files import Url
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
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)
from huggingface_hub.hf_api import CommitInfo, DatasetInfo, HfApi, RepoFile
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
    DatasetTooBigFromDatasetsError,
    DatasetTooBigFromHubError,
    DatasetWithTooBigExternalFilesError,
    DatasetWithTooManyExternalFilesError,
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
from worker.dtos import CompleteJobResult, ConfigParquetAndInfoResponse, ParquetFileItem
from worker.job_runners.config.config_job_runner import ConfigJobRunnerWithDatasetsCache
from worker.utils import retry, create_branch

DATASET_TYPE = "dataset"
MAX_FILES_PER_DIRECTORY = 10_000  # hf hub limitation
MAX_OPERATIONS_PER_COMMIT = 500


class ParquetFile:
    def __init__(self, local_file: str, local_dir: str, config: str):
        if not local_file.startswith(local_dir):
            raise ValueError(f"{local_file} is not in {local_dir}")
        self.local_file = local_file
        self.local_dir = local_dir
        self.config = config

    @property
    def path_in_repo(self) -> str:
        return f'{self.config}/{self.local_file.removeprefix(f"{self.local_dir}/")}'


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
    Raises the following errors:
        - [`libcommon.exceptions.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
    """
    if dataset in blocked_datasets:
        raise DatasetInBlockListError(
            "The parquet conversion has been disabled for this dataset for now. Please open an issue in"
            " https://github.com/huggingface/datasets-server if you want this dataset to be supported."
        )


def is_parquet_builder_with_hub_files(builder: DatasetBuilder, hf_endpoint: str) -> bool:
    if not isinstance(builder, ParquetBuilder) or not builder.config.data_files:
        return False
    for split in builder.config.data_files:
        for data_file in builder.config.data_files[split]:
            if not isinstance(data_file, Url):
                return False
            if not data_file.startswith(hf_endpoint + "/datasets/" + str(builder.repo_id) + "/"):
                return False
    return True


def raise_if_too_big_from_hub(
    dataset_info: DatasetInfo,
    max_dataset_size: int,
) -> None:
    """
    Raise an error if the dataset is too big to be converted to parquet, as measured by the sum of the repository
    files sizes given by the Hub.

    Args:
        dataset_info (`DatasetInfo`):
            The dataset info
        max_dataset_size (`int`):
            The maximum size of the dataset in bytes
    Returns:
        `None`
    Raises the following errors:
        - [`libcommon.exceptions.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet, as measured by the sum of the repository
          files sizes given by the Hub.
    """
    dataset_size: int = sum(sibling.size for sibling in dataset_info.siblings if sibling.size is not None)
    if dataset_size > max_dataset_size:
        raise DatasetTooBigFromHubError(
            f"The conversion to parquet is limited to datasets under {max_dataset_size} bytes. "
            f"Current size of files on the hub is {dataset_size} bytes."
        )


def raise_if_too_big_from_datasets(
    info: datasets.DatasetInfo,
    max_dataset_size: int,
) -> None:
    """
    Raise an error if the dataset is too big to be converted to parquet, as measured by the sum of the configs
    sizes given by the datasets library

    Args:
        info (`datasets.DatasetInfo`):
            Dataset info from the datasets library
        max_dataset_size (`int`):
            The maximum size of the dataset in bytes
    Returns:
        `None`
    Raises the following errors:
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`libcommon.exceptions.DatasetTooBigFromDatasetsError`]
          If the dataset is too big to be converted to parquet, as measured by the sum of the configs
          sizes given by the datasets library.
    """
    dataset_size = info.dataset_size if info.dataset_size is not None else 0
    if dataset_size > max_dataset_size:
        raise DatasetTooBigFromDatasetsError(
            f"The dataset is too big to be converted to Parquet. The size of the dataset ({dataset_size} B, as given"
            f" per the datasets library) exceeds the maximum supported size ({max_dataset_size} B). Please report the"
            " issue."
        )


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
            StreamingDownloadManager(
                base_path=builder.base_path, download_config=DownloadConfig(use_auth_token=hf_token)
            )
        )
    except ManualDownloadError as err:
        raise DatasetManualDownloadError(f"dataset={builder.repo_id} requires manual download.", cause=err) from err


def raise_if_not_supported(
    dataset_info: DatasetInfo,
    builder: DatasetBuilder,
    hf_endpoint: str,
    hf_token: Optional[str],
    max_dataset_size: int,
    max_external_data_files: int,
) -> None:
    """
    Raise an error if the dataset is not supported:
    - if the dataset is in the list of blocked datasets
    - if the dataset cannot be accessed (does not exist, private)
    - if the dataset is too big, and not in the list of supported datasets

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
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          If the datasets.config.HF_ENDPOINT is not set to the expected value
    """
    if datasets.config.HF_ENDPOINT != hf_endpoint:
        raise ValueError(
            f"Invalid datasets.config.HF_ENDPOINT value: '{datasets.config.HF_ENDPOINT}'. Please set it to:"
            f" '{hf_endpoint}'."
        )
    raise_if_requires_manual_download(
        builder=builder,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
    )
    raise_if_too_big_from_hub(dataset_info=dataset_info, max_dataset_size=max_dataset_size)
    raise_if_too_big_from_external_data_files(
        builder=builder,
        max_dataset_size=max_dataset_size,
        max_external_data_files=max_external_data_files,
        hf_token=hf_token,
    )
    raise_if_too_big_from_datasets(
        builder.info,
        max_dataset_size=max_dataset_size,
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


def get_writer_batch_size(ds_config_info: datasets.info.DatasetInfo) -> Optional[int]:
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


def copy_parquet_files(builder: DatasetBuilder) -> List[CommitOperationCopy]:
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
        num_shards = len(data_files[split])
        for shard_idx, data_file in enumerate(data_files[split]):
            src_revision, src_path_in_repo = data_file.split("/datasets/" + builder.repo_id + "/resolve/", 1)[1].split(
                "/", 1
            )

            # for forward compatibility with https://github.com/huggingface/datasets/pull/5331
            parquet_name = str(builder.dataset_name) if hasattr(builder, "dataset_name") else builder.name

            if num_shards > 1:
                path_in_repo = (
                    f"{builder.config.name}/{parquet_name}-{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
                )
            else:
                path_in_repo = f"{builder.config.name}/{parquet_name}-{split}.parquet"

            parquet_operations.append(
                CommitOperationCopy(
                    src_path_in_repo=src_path_in_repo, path_in_repo=path_in_repo, src_revision=src_revision
                )
            )
    return parquet_operations


class NotAParquetFileError(ValueError):
    """When a remote parquet file can't be parsed"""

    pass


def get_parquet_file_and_size(url: str, fs: HTTPFileSystem, hf_token: Optional[str]) -> Tuple[pq.ParquetFile, int]:
    headers = get_authentication_headers_for_url(url, use_auth_token=hf_token)
    f = fs.open(url, headers=headers)
    return pq.ParquetFile(f), f.size


def retry_get_parquet_file_and_size(
    url: str, fs: HTTPFileSystem, hf_token: Optional[str]
) -> Tuple[pq.ParquetFile, int]:
    try:
        sleeps = [1, 1, 1, 10, 10, 10]
        pf, size = retry(on=[pa.ArrowInvalid], sleeps=sleeps)(get_parquet_file_and_size)(url, fs, hf_token)
        return pf, size
    except RuntimeError as err:
        if err.__cause__ and isinstance(err.__cause__, pa.ArrowInvalid):
            raise NotAParquetFileError(f"Not a parquet file: '{url}'") from err.__cause__
        else:
            raise err


def fill_builder_info(builder: DatasetBuilder, hf_token: Optional[str]) -> None:
    """Fill the builder DatasetInfo from the copied parquet files"""
    data_files = builder.config.data_files
    if not data_files:
        raise EmptyDatasetError("Empty parquet data_files")
    fs = HTTPFileSystem()
    if not builder.info.splits or not builder.info.download_size:
        builder.info.splits = SplitDict()
        builder.info.download_size = 0
        builder.info.dataset_size = 0
        for split in data_files:
            split = str(split)  # in case it's a NamedSplit
            try:
                parquet_files_and_sizes: List[Tuple[pq.ParquetFile, int]] = thread_map(
                    partial(retry_get_parquet_file_and_size, fs=fs, hf_token=hf_token),
                    data_files[split],
                    unit="pq",
                    disable=True,
                )
                parquet_files, sizes = zip(*parquet_files_and_sizes)
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


def convert_to_parquet(builder: DatasetBuilder) -> List[CommitOperationAdd]:
    """Download and prepare the dataset as parquet files and fills the builder info"""
    # prepare the parquet files locally
    writer_batch_size = get_writer_batch_size(builder.info)
    if writer_batch_size is not None and (
        builder._writer_batch_size is None or builder._writer_batch_size > writer_batch_size
    ):
        builder._writer_batch_size = writer_batch_size
    builder.download_and_prepare(
        file_format="parquet"
    )  # the parquet files are stored in the cache dir and it fills the info
    local_parquet_files = [
        ParquetFile(local_file=local_file, local_dir=builder.cache_dir, config=builder.config.name)
        for local_file in glob.glob(f"{builder.cache_dir}**/*.parquet")
    ]

    # send the files to the target revision
    parquet_operations: List[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo=parquet_file.path_in_repo, path_or_fileobj=parquet_file.local_file)
        for parquet_file in local_parquet_files
    ]
    logging.debug(f"{parquet_operations=}")
    return parquet_operations


def create_commits(
    hf_api: HfApi,
    repo_id: str,
    operations: List[CommitOperation],
    *,
    commit_message: str,
    revision: Optional[str] = None,
    parent_commit: Optional[str] = None,
    max_operations_per_commit: int = MAX_OPERATIONS_PER_COMMIT,
) -> List[CommitInfo]:
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
        [`List[huggingface_hub.CommitInfo]`]:
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
    commit_infos: List[CommitInfo] = []
    sleeps = [1, 1, 1, 10, 10, 10]
    offsets = range(0, len(operations), max_operations_per_commit)
    for commit_idx, offset in enumerate(offsets):
        batch_msg = f" (step {commit_idx + 1} of {len(offsets)})" if len(offsets) > 1 else ""
        retry_create_commit = retry(on=[HfHubHTTPError], sleeps=sleeps)(hf_api.create_commit)
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
                        f" {len(sleeps)} attempts)."
                    ),
                    cause=e.__cause__,
                ) from e.__cause__
            raise e
        commit_infos.append(commit_info)
    return commit_infos


def get_delete_operations(
    parquet_operations: List[CommitOperationAdd], all_repo_files: Set[str], config_names: Set[str], config: str
) -> List[CommitOperationDelete]:
    # - get files that will be preserved in repo:
    #   1. parquet files belonging to any other config (otherwise outdated files might be preserved)
    #   2. duckdb files belonging to any config
    #   3. .gitattributes
    pattern_in_any_config_dir = re.compile(f"^({'|'.join(re.escape(conf) for conf in config_names)})/")
    pattern_in_any_other_config_dir = re.compile(
        f"^({'|'.join(re.escape(conf) for conf in config_names.difference({config}))})/"
    )
    files_to_ignore: Set[str] = {
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
    config_names: Set[str],
    parquet_operations: List[CommitOperation],
    commit_message: str,
    target_revision: Optional[str],
) -> List[CommitInfo]:
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
        config_names (`List[str]`):
            The list of all the configurations of this dataset. This is used to clean
            the other fiels and directories in the repo, if any.
        parquet_operations (`List[huggingface_hub.hf_api.CommitOperation]`):
            List of commit operation for the parquet conversion. It could be
            file additions or file copies for example.
        commit_message (`str`):
            The summary (first line) of the commit that will be created.
        target_revision (`str`, *optional*):
            The git revision to commit from. Defaults to the head of the `"main"` branch.
    Returns:
        [`List[huggingface_hub.CommitInfo]`]:
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
    all_repo_files: Set[str] = {f.rfilename for f in target_dataset_info.siblings}
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
    supported_datasets: List[str],
    blocked_datasets: List[str],
    max_dataset_size: int,
    max_external_data_files: int,
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
            use_auth_token=hf_token,
            download_config=download_config,
        )
    except _EmptyDatasetError as err:
        raise EmptyDatasetError(f"{dataset=} is empty.", cause=err) from err
    except FileNotFoundError as err:
        raise DatasetNotFoundError("The dataset, or the revision, does not exist on the Hub.") from err

    if is_parquet_builder_with_hub_files(builder, hf_endpoint=hf_endpoint):
        parquet_operations = copy_parquet_files(builder)
        fill_builder_info(builder, hf_token=hf_token)
    else:
        dataset_info = get_dataset_info_for_supported_datasets(
            dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, revision=source_revision, files_metadata=True
        )
        if dataset not in supported_datasets:
            raise_if_not_supported(
                dataset_info=dataset_info,
                builder=builder,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                max_dataset_size=max_dataset_size,
                max_external_data_files=max_external_data_files,
            )
        parquet_operations = convert_to_parquet(builder)

    try:
        sleeps = [1, 1, 1, 1, 1, 10, 10, 10, 10, 100] * 3
        # ^ timeouts after ~7 minutes
        with lock.git_branch(dataset=dataset, branch=target_revision, job_id=job_id, sleeps=sleeps):
            # create the target revision if we managed to get the parquet files and it does not exist yet
            # (clone from initial commit to avoid cloning all repo's files)
            refs = retry(on=[requests.exceptions.ConnectionError], sleeps=[1, 1, 1, 10, 10])(hf_api.list_repo_refs)(
                repo_id=dataset, repo_type=DATASET_TYPE
            )
            create_branch(
                dataset=dataset,
                target_revision=target_revision,
                refs=refs,
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
                supported_datasets=self.parquet_and_info_config.supported_datasets,
                blocked_datasets=self.parquet_and_info_config.blocked_datasets,
                max_dataset_size=self.parquet_and_info_config.max_dataset_size,
                max_external_data_files=self.parquet_and_info_config.max_external_data_files,
            )
        )
