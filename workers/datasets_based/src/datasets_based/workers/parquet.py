# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import glob
import importlib.metadata
import logging
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict
from urllib.parse import quote

from datasets import get_dataset_config_names, load_dataset_builder
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from huggingface_hub.hf_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    RepoFile,
)
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from libcommon.exceptions import CustomError
from libcommon.worker import DatasetNotFoundError, Worker

from datasets_based.config import AppConfig, ParquetConfig

ParquetWorkerErrorCode = Literal[
    "DatasetRevisionNotFoundError",
    "EmptyDatasetError",
    "ConfigNamesError",
    "DatasetNotSupportedError",
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


class DatasetNotSupportedError(ParquetWorkerError):
    """Raised when the dataset is not in the list of supported datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetNotSupportedError", cause, False)


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


# TODO: use huggingface_hub's hf_hub_url after
# https://github.com/huggingface/huggingface_hub/issues/1082
def hf_hub_url(repo_id: str, filename: str, hf_endpoint: str, revision: str, url_template: str) -> str:
    return (hf_endpoint + url_template).format(
        repo_id=repo_id,
        revision=quote(revision, safe=""),
        filename=filename,
    )


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


def compute_parquet_response(
    dataset: str,
    hf_endpoint: str,
    hf_token: str,
    source_revision: str,
    target_revision: str,
    commit_message: str,
    url_template: str,
    supported_datasets: List[str],
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
        hf_token (`str`):
            An authentication token (See https://huggingface.co/settings/token). It must:
            - be a user token (to get access to the gated datasets, and do the other operations)
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
    Returns:
        `ParquetResponseResult`: An object with the parquet_response
          (dataset and list of parquet files) and the dataset_git_revision (sha) if any.
    <Tip>
    Raises the following errors:
        - [`~parquet.worker.GatedExtraFieldsError`]
          If the dataset is gated and has "extra fields". This is not supported at the moment.
        - [`~parquet.worker.GatedDisabledError`]
          If the dataset is gated and the access is disabled.
        - [`~libcommon.worker.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~parquet.worker.DatasetRevisionNotFoundError`]
          If the revision does not exist or cannot be accessed using the token.
        - [`~splits.worker.EmptyDatasetError`]
          The dataset is empty.
        - [`~splits.worker.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get splits for dataset={dataset}")

    # only process the supported datasets
    if len(supported_datasets) and dataset not in supported_datasets:
        raise DatasetNotSupportedError("The dataset is not in the list of supported datasets.")

    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)

    # check that the revision exists for the dataset and is accessible using the token
    try:
        hf_api.dataset_info(repo_id=dataset, revision=source_revision, files_metadata=False)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    except RevisionNotFoundError as err:
        raise DatasetRevisionNotFoundError("The dataset revision does not exist on the Hub.") from err

    # create the target revision if it does not exist yet
    try:
        target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    except RevisionNotFoundError:
        # create the parquet_ref (refs/convert/parquet)
        hf_api.create_branch(repo_id=dataset, branch=target_revision, repo_type=DATASET_TYPE)
        target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)

    target_sha = target_dataset_info.sha
    previous_files = [f.rfilename for f in target_dataset_info.siblings]

    # get the sorted list of configurations
    try:
        config_names = sorted(get_dataset_config_names(path=dataset, use_auth_token=hf_token))
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

    # send the files to the target revision
    files_to_add = {parquet_file.repo_file(): parquet_file.local_file for parquet_file in parquet_files}
    # don't delete the files we will update
    files_to_delete = [file for file in previous_files if file not in files_to_add]
    delete_operations: List[CommitOperation] = [CommitOperationDelete(path_in_repo=file) for file in files_to_delete]
    add_operations: List[CommitOperation] = [
        CommitOperationAdd(path_in_repo=file, path_or_fileobj=local_file)
        for (file, local_file) in files_to_add.items()
    ]
    hf_api.create_commit(
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


class ParquetWorker(Worker):
    parquet_config: ParquetConfig

    def __init__(self, app_config: AppConfig, endpoint: str):
        super().__init__(
            processing_step=app_config.processing_graph.graph.get_step(endpoint),
            # ^ raises if the step is not found
            common_config=app_config.common,
            queue_config=app_config.queue,
            worker_config=app_config.worker,
            version=importlib.metadata.version(__package__.split(".")[0]),
        )
        self.parquet_config = app_config.parquet

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
            hf_token=self.parquet_config.hf_token,
            source_revision=self.parquet_config.source_revision,
            target_revision=self.parquet_config.target_revision,
            commit_message=self.parquet_config.commit_message,
            url_template=self.parquet_config.url_template,
            supported_datasets=self.parquet_config.supported_datasets,
        )
