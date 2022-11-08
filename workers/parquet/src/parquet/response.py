# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import glob
import logging
from pathlib import Path
from typing import List, Optional, TypedDict, Union, cast
from urllib.parse import quote

from datasets import get_dataset_config_names, load_dataset_builder
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from huggingface_hub.hf_api import (  # type: ignore
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
)
from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore

from parquet.utils import ConfigNamesError, DatasetNotFoundError, EmptyDatasetError


class ParquetFileItem:
    url: str
    filename: str
    size: int


class ParquetResponse(TypedDict):
    dataset: str
    parquet_files: List[ParquetFileItem]


class ParquetResponseResult(TypedDict):
    parquet_response: ParquetResponse
    dataset_git_revision: Optional[str]


def get_dataset_git_revision(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> Union[str, None]:
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
        - [`~worker.exceptions.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
    </Tip>
    """
    try:
        dataset_info = HfApi(endpoint=hf_endpoint).dataset_info(repo_id=dataset, token=hf_token)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    return dataset_info.sha


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


def compute_parquet_response(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    source_revision: str,
    target_revision: str,
    commit_message: str,
    url_template: str,
) -> ParquetResponseResult:
    """
    Get the response of /parquet for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        source_revision (`str`):
            The git revision (sha) of the dataset used to prepare the parquet files
        target_revision (`str`):
            The target git revision (sha) of the dataset where to store the parquet files
        commit_message (`str`):
            The commit message to use when storing the parquet files
        url_template (`str`):
            The template to use to build the parquet file url
    Returns:
        `ParquetResponseResult`: An object with the parquet_response
          (dataset and list of parquet files) and the dataset_git_revision (sha) if any.
    <Tip>
    Raises the following errors:
        - [`~worker.exceptions.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~worker.exceptions.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get splits for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # first try to get the dataset config info. It raises if the dataset does not exist or is private
    dataset_git_revision = get_dataset_git_revision(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token)
    # get the sorted list of configurations
    try:
        config_names = sorted(get_dataset_config_names(path=dataset, use_auth_token=use_auth_token))
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the configuration names for the dataset.", cause=err) from err

    # for simplicity, we first compute all the parquet files and store them locally, then we upload them to the hub
    # as datasets does not give the list of files per split, we don't retro-engineer the split names
    # and just avoid associating a split to a parquet file

    # first get the current sha (to be able to use parent_commit in create_commit)
    hf_api = HfApi(endpoint=hf_endpoint)

    # We assume that the parquet_ref (refs/convert/parquet) already exists.
    # See https://github.com/huggingface/huggingface_hub/issues/1165

    dataset_info = hf_api.dataset_info(repo_id=dataset, revision=target_revision, files_metadata=False)
    parent_commit = dataset_info.sha
    previous_files = [f.rfilename for f in dataset_info.siblings]

    parquet_files: List[ParquetFile] = []
    for config in config_names:
        builder = load_dataset_builder(path=dataset, name=config, revision=source_revision)
        builder.download_and_prepare(file_format="parquet")  # the parquet files are stored in the cache dir
        parquet_files.extend(
            ParquetFile(local_file=local_file, local_dir=builder.cache_dir, config=config)
            for local_file in glob.glob(f"{builder.cache_dir}**/*.parquet")
        )

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
        token=hf_token,
        repo_type="dataset",
        revision=target_revision,
        operations=delete_operations + add_operations,
        commit_message=commit_message,
        parent_commit=parent_commit,
    )

    # call the API to get the list of parquet files
    repo_files = [
        repo_file
        for repo_file in hf_api.dataset_info(
            repo_id=dataset, revision=target_revision, files_metadata=True, token=hf_token
        ).siblings
        if repo_file.rfilename.endswith(".parquet")
    ]

    return {
        "parquet_response": {
            "dataset": dataset,
            "parquet_files": [
                cast(
                    ParquetFileItem,
                    {
                        "url": hf_hub_url(
                            repo_id=dataset,
                            filename=repo_file.rfilename,
                            hf_endpoint=hf_endpoint,
                            revision=target_revision,
                            url_template=url_template,
                        ),
                        "filename": Path(repo_file.rfilename).name,
                        "size": repo_file.size,
                    },
                )
                for repo_file in repo_files
            ],
        },
        "dataset_git_revision": dataset_git_revision,
    }
