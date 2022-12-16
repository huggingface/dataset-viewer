# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import contextlib
import glob
import logging
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict
from urllib.parse import quote

import datasets.config
from datasets import (
    get_dataset_config_info,
    get_dataset_config_names,
    load_dataset_builder,
)
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
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
from libcommon.exceptions import CustomError
from libcommon.worker import DatasetNotFoundError

from datasets_based.config import AppConfig, ParquetConfig
from datasets_based.workers._datasets_based_worker import DatasetsBasedWorker

ParquetWorkerErrorCode = Literal[
    "DatasetRevisionNotFoundError",
    "EmptyDatasetError",
    "ConfigNamesError",
    "DatasetInBlockListError",
    "DatasetTooBigFromHubError",
    "DatasetTooBigFromDatasetsError",
]


class ParquetWorkerError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ParquetWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class DatasetRevisionNotFoundError(ParquetWorkerError):
    """Raised when the revision of a dataset repository does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "DatasetRevisionNotFoundError", cause, False)


class ConfigNamesError(ParquetWorkerError):
    """Raised when the configuration names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ConfigNamesError", cause, True)


class EmptyDatasetError(ParquetWorkerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class DatasetInBlockListError(ParquetWorkerError):
    """Raised when the dataset is in the list of blocked datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetInBlockListError", cause, False)


class DatasetTooBigFromHubError(ParquetWorkerError):
    """Raised when the dataset size (sum of files on the Hub) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromHubError", cause, False)


class DatasetTooBigFromDatasetsError(ParquetWorkerError):
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


class ParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]


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


# until https://github.com/huggingface/datasets/pull/5333 is merged
def get_dataset_infos(path: str, revision: Optional[str] = None, use_auth_token: Optional[str] = None):
    """Get the meta information about a dataset, returned as a dict mapping config name to DatasetInfoDict.

    Args:
        path (``str``): a dataset identifier on the Hugging Face Hub (list all available datasets and ids with
            ``datasets.list_datasets()``)  e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``
        revision (Optional ``str``):
            If specified, the dataset module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the main branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        use_auth_token (``str``, optional): Optional string to use as Bearer token for remote files on the Datasets
            Hub.
    """
    config_names = get_dataset_config_names(
        path=path,
        revision=revision,
        use_auth_token=use_auth_token,
    )
    return {
        config_name: get_dataset_config_info(
            path=path, config_name=config_name, revision=revision, use_auth_token=use_auth_token
        )
        for config_name in config_names
    }


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
        - [`~parquet.worker.DatasetInBlockListError`]
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
        - [`~libcommon.worker.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~parquet.worker.DatasetRevisionNotFoundError`]
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
        - [`~parquet.worker.DatasetTooBigFromHubError`]
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
        - [`~parquet.worker.DatasetTooBigFromDatasetsError`]
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
        - [`~parquet.worker.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
        - [`~parquet.worker.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~parquet.worker.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
        - [`ValueError`]
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~parquet.worker.DatasetTooBigFromDatasetsError`]
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


def compute_parquet_response(
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
) -> ParquetResponse:
    """
    Get the response of /parquet for one specific dataset on huggingface.co.
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
    Returns:
        `ParquetResponseResult`: An object with the parquet_response
          (dataset and list of parquet files) and the dataset_git_revision (sha) if any.
    <Tip>
    Raises the following errors:
        - [`~parquet.worker.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
        - [`~libcommon.dataset.GatedExtraFieldsError`]: if the dataset is gated, with extra fields.
            Programmatic access is not implemented for this type of dataset because there is no easy
            way to get the list of extra fields.
        - [`~libcommon.dataset.GatedDisabledError`]: if the dataset is gated, but disabled.
        - [`~libcommon.dataset.DatasetNotFoundError`]: if the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
        - ['~requests.exceptions.HTTPError']: any other error when asking access
        - [`~parquet.worker.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~parquet.worker.DatasetTooBigFromHubError`]
          If the dataset is too big to be converted to parquet
        - [`ValueError`]
            If the datasets.config.HF_ENDPOINT is not set to the expected value
        - [`~parquet.worker.DatasetTooBigFromDatasetsError`]
            If the dataset is too big to be converted to parquet
        - [`~parquet.worker.EmptyDatasetError`]
          The dataset is empty.
        - [`~parquet.worker.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
        - [`~parquet.worker.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
    </Tip>
    """
    logging.info(f"get splits for dataset={dataset}")

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
            get_dataset_config_names(path=dataset, revision=source_revision, use_auth_token=hf_token)
        )
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the configuration names for the dataset.", cause=err) from err

    # prepare the parquet files locally
    parquet_files: List[ParquetFile] = []
    for config in config_names:
        builder = load_dataset_builder(path=dataset, name=config, revision=source_revision, use_auth_token=hf_token)
        builder.download_and_prepare(
            file_format="parquet", use_auth_token=hf_token
        )  # the parquet files are stored in the cache dir
        parquet_files.extend(
            ParquetFile(local_file=local_file, local_dir=builder.cache_dir, config=config)
            for local_file in glob.glob(f"{builder.cache_dir}**/*.parquet")
        )

    # create the target revision if it does not exist yet
    try:
        target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    except RevisionNotFoundError:
        # create the parquet_ref (refs/convert/parquet)
        committer_hf_api.create_branch(repo_id=dataset, branch=target_revision, repo_type=DATASET_TYPE)
        target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)

    target_sha = target_dataset_info.sha
    previous_files = [f.rfilename for f in target_dataset_info.siblings]

    # send the files to the target revision
    files_to_add = {parquet_file.repo_file(): parquet_file.local_file for parquet_file in parquet_files}
    # don't delete the files we will update
    files_to_delete = [file for file in previous_files if file not in files_to_add]
    delete_operations: List[CommitOperation] = [CommitOperationDelete(path_in_repo=file) for file in files_to_delete]
    add_operations: List[CommitOperation] = [
        CommitOperationAdd(path_in_repo=file, path_or_fileobj=local_file)
        for (file, local_file) in files_to_add.items()
    ]
    committer_hf_api.create_commit(
        repo_id=dataset,
        repo_type=DATASET_TYPE,
        revision=target_revision,
        operations=delete_operations + add_operations,
        commit_message=commit_message,
        parent_commit=target_sha,
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
    }


class ParquetWorker(DatasetsBasedWorker):
    parquet_config: ParquetConfig

    @staticmethod
    def get_endpoint() -> str:
        return "/parquet"

    def __init__(self, app_config: AppConfig):
        super().__init__(app_config=app_config)
        self.parquet_config = ParquetConfig()

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
    ) -> Mapping[str, Any]:
        return compute_parquet_response(
            dataset=dataset,
            hf_endpoint=self.common_config.hf_endpoint,
            hf_token=self.common_config.hf_token,
            committer_hf_token=self.parquet_config.committer_hf_token,
            source_revision=self.parquet_config.source_revision,
            target_revision=self.parquet_config.target_revision,
            commit_message=self.parquet_config.commit_message,
            url_template=self.parquet_config.url_template,
            supported_datasets=self.parquet_config.supported_datasets,
            blocked_datasets=self.parquet_config.blocked_datasets,
            max_dataset_size=self.parquet_config.max_dataset_size,
        )
