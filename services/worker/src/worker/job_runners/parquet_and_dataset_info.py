# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import glob
import logging
import re
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict
from urllib.parse import quote

import datasets
import datasets.config
import numpy as np
from datasets import get_dataset_config_names, get_dataset_infos, load_dataset_builder
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
from huggingface_hub.hf_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
    DatasetInfo,
    HfApi,
    RepoFile,
)
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from libcommon.dataset import ask_access
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import SplitFullName

from worker.config import AppConfig, ParquetAndDatasetInfoConfig
from worker.job_runner import DatasetNotFoundError, JobInfo, JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner

ParquetAndDatasetInfoJobRunnerErrorCode = Literal[
    "DatasetRevisionNotFoundError",
    "EmptyDatasetError",
    "ConfigNamesError",
    "DatasetInBlockListError",
    "DatasetTooBigFromHubError",
    "DatasetTooBigFromDatasetsError",
    "DatasetTooBigFromExternalFiles",
]


class ParquetAndDatasetInfoJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ParquetAndDatasetInfoJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class DatasetRevisionNotFoundError(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the revision of a dataset repository does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "DatasetRevisionNotFoundError", cause, False)


class ConfigNamesError(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the configuration names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ConfigNamesError", cause, True)


class EmptyDatasetError(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class DatasetInBlockListError(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the dataset is in the list of blocked datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetInBlockListError", cause, False)


class DatasetTooBigFromHubError(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the dataset size (sum of files on the Hub) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromHubError", cause, False)


class DatasetTooBigFromDatasetsError(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromDatasetsError", cause, False)


class ParquetFileItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class ParquetAndDatasetInfoResponse(TypedDict):
    parquet_files: List[ParquetFileItem]
    dataset_info: dict[str, Any]


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


p = re.compile(r"[\w]+-(?P<split>[\w]+?)(-[0-9]{5}-of-[0-9]{5})?.parquet")


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
    hf_endpoint: str,
    target_revision: str,
    url_template: str,
) -> ParquetFileItem:
    if repo_file.size is None:
        raise ValueError(f"Cannot get size of {repo_file.rfilename}")
    config, split = parse_repo_filename(repo_file.rfilename)
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
        - [`~job_runners.parquet_and_dataset_info.DatasetInBlockListError`]
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
        - [`~job_runners.parquet_and_dataset_info.DatasetRevisionNotFoundError`]
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
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromHubError`]
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
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromDatasetsError`]
            If the dataset is too big to be converted to parquet
    </Tip>
    """
    if datasets.config.HF_ENDPOINT != hf_endpoint:
        raise ValueError(
            "datasets.config.HF_ENDPOINT should have already been set to {hf_endpoint}. "
            f"Current value: {datasets.config.HF_ENDPOINT}. "
        )
    dataset_size = 0
    with contextlib.suppress(Exception):
        infos = get_dataset_infos(path=dataset, revision=revision, use_auth_token=hf_token)
        dataset_size = sum(value.dataset_size for value in infos.values() if value.dataset_size is not None)
    if dataset_size > max_dataset_size:
        raise DatasetTooBigFromDatasetsError(
            f"The conversion to parquet is limited to datasets under {max_dataset_size} bytes. "
            f"Current size as given per the datasets library is {dataset_size} bytes."
        )


def raise_if_not_supported(
    dataset: str,
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
        - [`~job_runners.parquet_and_dataset_info.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
        - [`~job_runners.parquet_and_dataset_info.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
        - [`ValueError`]
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromDatasetsError`]
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


class DatasetTooBigFromExternalFiles(ParquetAndDatasetInfoJobRunnerError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromExternalFiles", cause, False)


def _request_size(url: str, hf_token: Optional[str] = None) -> Optional[int]:
    headers = get_authentication_headers_for_url(url, use_auth_token=hf_token)
    response = http_head(url, headers=headers, max_retries=3)
    response.raise_for_status()
    size = response.headers.get("Content-Length") if response.ok else None
    return int(size) if size is not None else size


class _MockStreamingDownloadManager(StreamingDownloadManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_data_files = []

    def download(self, url_or_urls):
        url_or_urls = map_nested(self._download, url_or_urls, map_tuple=True, parallel_min_length=np.inf)
        return url_or_urls

    def _download(self, urlpath):
        urlpath = str(urlpath)
        if is_relative_path(urlpath):
            # append the relative path to the base_path
            urlpath = url_or_path_join(self._base_path, urlpath)
        elif not urlpath.startswith(self._base_path):
            # it's an external file
            self.ext_data_files.append(urlpath)
        return urlpath


def raise_if_too_big_from_external_data_files(
    builder: DatasetBuilder, max_dataset_size: int, max_external_data_files: int, hf_token: Optional[str]
) -> None:
    # Packaged dataset modules only load data files that are inside the dataset repository.
    # No need to check them since they're already caught by `raise_if_too_big_from_hub`
    if type(builder).__module__.startswith("datasets."):
        return
    # For datasets with a loading script however, we need to check the downloaded files
    mock_dl_manager = _MockStreamingDownloadManager(base_path=builder.base_path)
    try:
        builder._split_generators(mock_dl_manager)
    except NotImplementedError as err:
        if "is not implemented in streaming mode." not in str(err):
            raise
    ext_data_files = mock_dl_manager.ext_data_files
    if len(ext_data_files) > max_external_data_files:
        raise DatasetTooBigFromExternalFiles(
            f"The conversion to parquet is limited to datasets with less than {max_external_data_files} files. "
            f"However it uses {len(ext_data_files)} data files."
        )
    elif ext_data_files:
        from multiprocessing.pool import ThreadPool

        with ThreadPool(16) as pool:
            total_size = 0
            get_size = partial(_request_size, hf_token=hf_token)
            for i, size in enumerate(pool.imap_unordered(get_size, ext_data_files)):
                if size is not None:
                    total_size += size
                    if total_size > max_dataset_size:
                        raise DatasetTooBigFromExternalFiles(
                            f"The conversion to parquet is limited to datasets under {max_dataset_size} bytes. However"
                            f" {i + 1} data files of {len(ext_data_files)} are already bigger than {total_size} bytes."
                        )


def compute_parquet_and_dataset_info_response(
    dataset: str,
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
) -> ParquetAndDatasetInfoResponse:
    """
    Get the response of /parquet-and-dataset-info for one specific dataset on huggingface.co.
    It is assumed that the dataset can be accessed with the token.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
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
        `ParquetAndDatasetInfoResponse`: An object with the parquet_and_dataset_info_response
          (dataset info and list of parquet files).
    <Tip>
    Raises the following errors:
        - [`~job_runners.parquet_and_dataset_info.DatasetInBlockListError`]
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
        - [`~job_runners.parquet_and_dataset_info.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromDatasetsError`]
            If the dataset is too big to be converted to parquet
        - [`~job_runners.parquet_and_dataset_info.EmptyDatasetError`]
          The dataset is empty.
        - [`~job_runners.parquet_and_dataset_info.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
        - [`~job_runners.parquet_and_dataset_info.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`~job_runners.parquet_and_dataset_info.DatasetTooBigFromExternalFiles`]
            If the dataset is too big to be converted to parquet
    </Tip>
    """
    logging.info(f"get parquet files and dataset info for dataset={dataset}")

    raise_if_not_supported(
        dataset=dataset,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        committer_hf_token=committer_hf_token,
        revision=source_revision,
        supported_datasets=supported_datasets,
        blocked_datasets=blocked_datasets,
        max_dataset_size=max_dataset_size,
    )

    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)
    committer_hf_api = HfApi(endpoint=hf_endpoint, token=committer_hf_token)

    # get the sorted list of configurations
    try:
        config_names = sorted(
            str(config)
            for config in get_dataset_config_names(path=dataset, revision=source_revision, use_auth_token=hf_token)
        )
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the configuration names for the dataset.", cause=err) from err

    # prepare the parquet files locally
    parquet_files: List[ParquetFile] = []
    dataset_info: dict[str, Any] = {}
    for config in config_names:
        builder = load_dataset_builder(path=dataset, name=config, revision=source_revision, use_auth_token=hf_token)
        raise_if_too_big_from_external_data_files(
            builder=builder, max_dataset_size=max_dataset_size, max_external_data_files=max_external_data_files
        )
        builder.download_and_prepare(file_format="parquet")  # the parquet files are stored in the cache dir
        dataset_info[config] = asdict(builder.info)
        # ^ see
        # https://github.dev/huggingface/datasets/blob/e183a269067575db8765ee979bd8523d14a1adae/src/datasets/info.py#L244-L245
        parquet_files.extend(
            ParquetFile(local_file=local_file, local_dir=builder.cache_dir, config=config)
            for local_file in glob.glob(f"{builder.cache_dir}**/*.parquet")
        )

    # create the target revision if it does not exist yet
    try:
        refs = hf_api.list_repo_refs(repo_id=dataset, repo_type=DATASET_TYPE)
        if all(ref.ref != target_revision for ref in refs.converts):
            committer_hf_api.create_branch(
                repo_id=dataset, branch=target_revision, repo_type=DATASET_TYPE, revision=source_revision
            )
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err

    # delete:
    # - the previous files,
    target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
    previous_files = {f.rfilename for f in target_dataset_info.siblings}
    # except:
    # - the files we will update,
    files_to_add = {parquet_file.repo_file(): parquet_file.local_file for parquet_file in parquet_files}
    # - .gitattributes if present.
    files_to_delete = previous_files - set(files_to_add.keys()).union({".gitattributes"})
    delete_operations: List[CommitOperation] = [CommitOperationDelete(path_in_repo=file) for file in files_to_delete]
    logging.debug(f"delete_operations={delete_operations}")

    # send the files to the target revision
    add_operations: List[CommitOperation] = [
        CommitOperationAdd(path_in_repo=file, path_or_fileobj=local_file)
        for (file, local_file) in files_to_add.items()
    ]
    logging.debug(f"add_operations={add_operations}")

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
    repo_files = [repo_file for repo_file in target_dataset_info.siblings if repo_file.rfilename.endswith(".parquet")]
    # we might want to check if the sha of the parquet files is the same as the one we just uploaded
    # we could also check that the list of parquet files is exactly what we expect
    # let's not over engineer this for now. After all, what is on the Hub is the source of truth
    # and the /parquet response is more a helper to get the list of parquet files
    return {
        "parquet_files": [
            create_parquet_file_item(
                repo_file=repo_file,
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                target_revision=target_revision,
                url_template=url_template,
            )
            for repo_file in repo_files
        ],
        "dataset_info": dataset_info,
    }


class ParquetAndDatasetInfoJobRunner(DatasetsBasedJobRunner):
    parquet_and_dataset_info_config: ParquetAndDatasetInfoConfig

    @staticmethod
    def get_job_type() -> str:
        return "/parquet-and-dataset-info"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
        parquet_and_dataset_info_config: ParquetAndDatasetInfoConfig,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.parquet_and_dataset_info_config = parquet_and_dataset_info_config

    def compute(self) -> Mapping[str, Any]:
        return compute_parquet_and_dataset_info_response(
            dataset=self.dataset,
            hf_endpoint=self.common_config.hf_endpoint,
            hf_token=self.common_config.hf_token,
            committer_hf_token=self.parquet_and_dataset_info_config.committer_hf_token,
            source_revision=self.parquet_and_dataset_info_config.source_revision,
            target_revision=self.parquet_and_dataset_info_config.target_revision,
            commit_message=self.parquet_and_dataset_info_config.commit_message,
            url_template=self.parquet_and_dataset_info_config.url_template,
            supported_datasets=self.parquet_and_dataset_info_config.supported_datasets,
            blocked_datasets=self.parquet_and_dataset_info_config.blocked_datasets,
            max_dataset_size=self.parquet_and_dataset_info_config.max_dataset_size,
        )

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=parquet_file["dataset"], config=parquet_file["config"], split=parquet_file["split"])
            for parquet_file in content["parquet_files"]
        }
