# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import itertools
import logging
import sys
import traceback
import warnings
from fnmatch import fnmatch
from typing import Optional, Union
from urllib.parse import quote

import PIL
import requests
from datasets import Dataset, DatasetInfo, DownloadConfig, IterableDataset, load_dataset
from datasets.utils.file_utils import get_authentication_headers_for_url
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from libcommon.constants import CONFIG_SPLIT_NAMES_KIND, EXTERNAL_DATASET_SCRIPT_PATTERN
from libcommon.dtos import RowsContent
from libcommon.exceptions import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    DatasetWithScriptNotSupportedError,
    NormalRowsError,
    PreviousStepFormatError,
    SplitNotFoundError,
    StreamingRowsError,
)
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import retry
from pyarrow.parquet import ParquetFile

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


def get_parquet_file(url: str, fs: HTTPFileSystem, hf_token: Optional[str]) -> ParquetFile:
    headers = get_authentication_headers_for_url(url, token=hf_token)
    return ParquetFile(fs.open(url, headers=headers))


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
        if (allowed_pattern == "{{ALL_DATASETS_WITH_NO_NAMESPACE}}" and "/" not in dataset) or fnmatch(
            dataset, allowed_pattern
        ):
            return True
    return False
