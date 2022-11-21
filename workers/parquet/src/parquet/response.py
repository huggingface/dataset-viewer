# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import glob
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Union
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

from parquet.utils import ConfigNamesError, DatasetNotFoundError, EmptyDatasetError


class ParquetFileItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class ParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]


class ParquetResponseResult(TypedDict):
    parquet_response: ParquetResponse
    dataset_git_revision: Optional[str]


DATASET_TYPE = "dataset"


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
        - [`~worker.exceptions.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~worker.exceptions.ConfigNamesError`]
          If the list of configurations could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get splits for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False

    hf_api = HfApi(endpoint=hf_endpoint, token=hf_token)

    # get the SHA of the source revision
    try:
        source_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=source_revision, files_metadata=False)
        source_sha = source_dataset_info.sha
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    except RevisionNotFoundError as err:
        raise DatasetNotFoundError("The dataset revision does not exist on the Hub.") from err

    # create the target revision if it does not exist yet
    try:
        target_dataset_info = hf_api.dataset_info(repo_id=dataset, revision=source_revision, files_metadata=False)
        target_sha = target_dataset_info.sha
        previous_files = [f.rfilename for f in target_dataset_info.siblings]
    except RevisionNotFoundError:
        # create the parquet_ref (refs/convert/parquet)
        hf_api.create_branch(repo_id=dataset, branch="refs/convert/parquet", repo_type=DATASET_TYPE)

    # get the sorted list of configurations
    try:
        config_names = sorted(get_dataset_config_names(path=dataset, use_auth_token=use_auth_token))
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the configuration names for the dataset.", cause=err) from err

    # prepare the parquet files locally
    parquet_files: List[ParquetFile] = []
    for config in config_names:
        builder = load_dataset_builder(path=dataset, name=config, revision=source_revision)
        builder.download_and_prepare(file_format="parquet")  # the parquet files are stored in the cache dir
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
        "parquet_response": {
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
        },
        "dataset_git_revision": source_sha,
    }
