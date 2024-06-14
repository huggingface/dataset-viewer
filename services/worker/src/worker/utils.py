# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import itertools
import logging
import os
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Optional, Union
from urllib.parse import quote

import PIL
import requests
from datasets import Dataset, DatasetInfo, DownloadConfig, Features, IterableDataset, load_dataset
from datasets.utils.file_utils import SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL
from huggingface_hub import HfFileSystem, HfFileSystemFile
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from libcommon.constants import CONFIG_SPLIT_NAMES_KIND, EXTERNAL_DATASET_SCRIPT_PATTERN, MAX_COLUMN_NAME_LENGTH
from libcommon.dtos import RowsContent
from libcommon.exceptions import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    DatasetWithScriptNotSupportedError,
    NormalRowsError,
    PreviousStepFormatError,
    SplitNotFoundError,
    StreamingRowsError,
    TooLongColumnNameError,
)
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import retry
from pyarrow import ArrowInvalid

MAX_IMAGE_PIXELS = 10_000_000_000
# ^ see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.MAX_IMAGE_PIXELS


@retry(on=[ConnectionError])
def get_rows(
    dataset: str,
    config: str,
    split: str,
    streaming: bool,
    rows_max_number: int,
    token: Union[bool, str, None] = False,
    column_names: Optional[list[str]] = None,
    trust_remote_code: bool = False,
) -> RowsContent:
    download_config = DownloadConfig(delete_extracted=True)
    PIL.Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    ds = load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=streaming,
        token=token,
        download_config=download_config,
        trust_remote_code=trust_remote_code,
    )
    if streaming:
        if not isinstance(ds, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset in streaming mode")
    elif not isinstance(ds, Dataset):
        raise TypeError("load_dataset should return a Dataset in normal mode")
    if column_names:
        ds = ds.select_columns(column_names)
    rows_plus_one = list(itertools.islice(ds, rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
    rows = rows_plus_one[:rows_max_number]
    all_fetched = len(rows_plus_one) <= rows_max_number
    if all_fetched:
        logging.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logging.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return RowsContent(rows=rows, all_fetched=all_fetched, truncated_columns=[])


def get_rows_or_raise(
    dataset: str,
    config: str,
    split: str,
    rows_max_number: int,
    token: Union[bool, str, None],
    info: DatasetInfo,
    max_size_fallback: Optional[int] = None,
    column_names: Optional[list[str]] = [],
    trust_remote_code: bool = False,
) -> RowsContent:
    try:
        return get_rows(
            dataset=dataset,
            config=config,
            split=split,
            streaming=True,
            rows_max_number=rows_max_number,
            token=token,
            column_names=column_names,
            trust_remote_code=trust_remote_code,
        )
    except Exception as err:
        if isinstance(err, ValueError) and "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError from err
        MAX_SIZE_FALLBACK = 100_000_000
        if max_size_fallback:
            warnings.warn(
                (
                    f"The parameter 'max_size_fallback' is deprecated. The hard-coded value `{MAX_SIZE_FALLBACK}`"
                    " will be used instead."
                ),
                category=DeprecationWarning,
            )
        if info.size_in_bytes is None or info.size_in_bytes > MAX_SIZE_FALLBACK:
            raise StreamingRowsError(
                "Cannot load the dataset split (in streaming mode) to extract the first rows.",
                cause=err,
            ) from err
        try:
            return get_rows(
                dataset=dataset,
                config=config,
                split=split,
                streaming=False,
                rows_max_number=rows_max_number,
                token=token,
            )
        except Exception as err:
            if isinstance(err, ValueError) and "trust_remote_code" in str(err):
                raise DatasetWithScriptNotSupportedError from err
            raise NormalRowsError(
                "Cannot load the dataset split (in normal download mode) to extract the first rows.",
                cause=err,
            ) from err


# TODO: use huggingface_hub's hf_hub_url after
# https://github.com/huggingface/huggingface_hub/issues/1082
def hf_hub_url(repo_id: str, filename: str, hf_endpoint: str, revision: str, url_template: str) -> str:
    return (hf_endpoint + url_template) % (repo_id, quote(revision, safe=""), filename)


def hffs_parquet_url(repo_id: str, config: str, split_directory: str, filename: str) -> str:
    """Construct url of a parquet file on the Hub, to be used with HfFileSystem."""
    return f"hf://datasets/{repo_id}/{config}/{split_directory}/{filename}"


def hf_hub_open_file(
    file_url: str, hf_endpoint: str, hf_token: Optional[str], revision: Optional[str] = None
) -> HfFileSystemFile:
    """Open file with the HfFileSystem."""
    fs = HfFileSystem(endpoint=hf_endpoint, token=hf_token)
    return fs.open(file_url, revision=revision)


# used by `config-parquet-and-info` and `config-parquet-metadata` steps
@retry(on=[ArrowInvalid], sleeps=[0.2, 1, 1, 10, 10, 10])
def retry_on_arrow_invalid_open_file(
    file_url: str, hf_endpoint: str, hf_token: Optional[str], revision: Optional[str] = None
) -> HfFileSystemFile:
    return hf_hub_open_file(file_url=file_url, hf_endpoint=hf_endpoint, hf_token=hf_token, revision=revision)


DATASET_TYPE = "dataset"

LIST_REPO_REFS_RETRY_SLEEPS = [1, 1, 1, 10, 10]
LOCK_GIT_BRANCH_RETRY_SLEEPS = [1, 1, 1, 1, 1, 10, 10, 10, 10, 100] * 3


def create_branch(dataset: str, target_revision: str, hf_api: HfApi, committer_hf_api: HfApi) -> None:
    try:
        refs = retry(on=[requests.exceptions.ConnectionError], sleeps=LIST_REPO_REFS_RETRY_SLEEPS)(
            hf_api.list_repo_refs
        )(repo_id=dataset, repo_type=DATASET_TYPE)
        if all(ref.ref != target_revision for ref in refs.converts):
            initial_commit = hf_api.list_repo_commits(repo_id=dataset, repo_type=DATASET_TYPE)[-1].commit_id
            committer_hf_api.create_branch(
                repo_id=dataset, branch=target_revision, repo_type=DATASET_TYPE, revision=initial_commit, exist_ok=True
            )
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub (was deleted during job).") from err


def check_config_exists(dataset: str, config: str) -> None:
    """
    Check if dataset has a provided config. Dataset's configs are taken from 'dataset-config-names' step's cache.
    """
    config_names_response = get_previous_step_or_raise(kind="dataset-config-names", dataset=dataset)
    try:
        configs_content = config_names_response["content"]["config_names"]
    except Exception as e:
        raise PreviousStepFormatError(
            "Previous steps 'dataset-config-names' did not return the expected content.",
            e,
        ) from e

    if config not in [config_item["config"] for config_item in configs_content]:
        raise ConfigNotFoundError(f"Config '{config}' does not exist for dataset '{dataset}'")


def check_split_exists(dataset: str, config: str, split: str) -> None:
    """
    Check if dataset has a provided split in a provided config. Dataset's splits are taken from 'config-split-names'
      step's cache.
    """
    check_config_exists(dataset, config)
    split_names_response = get_previous_step_or_raise(kind="config-split-names", dataset=dataset, config=config)
    try:
        splits_content = split_names_response["content"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError(
            "Previous step 'config-split-names' did not return" " the expected content.",
            e,
        ) from e

    if split not in [split_item["split"] for split_item in splits_content]:
        raise SplitNotFoundError(f"Split '{split}' does not exist for the config '{config}' of the dataset.")


def get_split_names(dataset: str, config: str) -> set[str]:
    split_names_response = get_previous_step_or_raise(kind=CONFIG_SPLIT_NAMES_KIND, dataset=dataset, config=config)

    split_names_content = split_names_response["content"]
    if "splits" not in split_names_content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    if not isinstance(split_names_content["splits"], list):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"'splits' should be a list, but got {type(split_names_content['splits'])}"),
        )
    return {split_name_item["split"] for split_name_item in split_names_content["splits"]}


def is_dataset_script_error() -> bool:
    (t, v, tb) = sys.exc_info()
    cause_traceback: list[str] = traceback.format_exception(t, v, tb)
    return any(EXTERNAL_DATASET_SCRIPT_PATTERN in cause for cause in cause_traceback)


def resolve_trust_remote_code(dataset: str, allow_list: list[str]) -> bool:
    for allowed_pattern in allow_list:
        if fnmatch(dataset, allowed_pattern):
            return True
    return False


def raise_if_long_column_name(features: Optional[Features]) -> None:
    if features is None:
        return
    for feature_name in features:
        if len(feature_name) > MAX_COLUMN_NAME_LENGTH:
            short_name = feature_name[: MAX_COLUMN_NAME_LENGTH - 3] + "..."
            raise TooLongColumnNameError(
                f"Column name '{short_name}' is too long. It should be less than {MAX_COLUMN_NAME_LENGTH} characters."
            )


FileExtensionTuple = tuple[str, Optional[str]]


@dataclass
class FileExtension:
    extension: str
    uncompressed_extension: Optional[str] = field(default=None)

    def get_tuples(self) -> list[FileExtensionTuple]:
        """
        Get the extension and the archived extension if it exists.

        The list contains two entries if the uncompressed extension exists (for the compressed and the uncompressed files),
          otherwise one entry.
        """
        if self.uncompressed_extension:
            return [
                (self.extension, None),
                (self.uncompressed_extension, self.extension),
            ]
        return [(self.extension, None)]


def get_file_extension(filename: str, recursive: bool = True, clean: bool = True) -> FileExtension:
    """
    Get the extension of a file.

    In the case of .tar.gz or other "double extensions", the uncompressed file extension is set in the uncompressed_extension field

    Args:
        filename (`str`): The name of the file.
        recursive (`bool`, *optional*): Whether to recursively extract the extension of the archive.
        clean (`bool`, *optional*): Whether to clean the extension by removing special characters.

    Returns:
        FileExtension: the extension of the file
    """
    [base, extension] = os.path.splitext(filename)
    extension = extension.lower()
    # special cases we find in datasets (gz?dl=1 -> gz, txt_1 -> txt, txt-00000-of-00100-> txt)
    # https://github.com/huggingface/datasets/blob/af3acfdfcf76bb980dbac871540e30c2cade0cf9/src/datasets/utils/file_utils.py#L795
    if clean:
        for symb in "?-_":
            extension = extension.split(symb)[0]
    if recursive and extension.lstrip(".") in SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL:
        uncompressed_extension = get_file_extension(base, recursive=False, clean=False)
        return FileExtension(extension=extension, uncompressed_extension=uncompressed_extension.extension)
    return FileExtension(extension=extension)
