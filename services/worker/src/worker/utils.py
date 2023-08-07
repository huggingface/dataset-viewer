# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import functools
import itertools
import logging
import time
import warnings
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import quote

import PIL
import requests
from datasets import Dataset, DatasetInfo, DownloadConfig, IterableDataset, load_dataset
from datasets.utils.file_utils import get_authentication_headers_for_url
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from libcommon.exceptions import (
    DatasetNotFoundError,
    NormalRowsError,
    PreviousStepFormatError,
    SplitNotFoundError,
    StreamingRowsError,
)
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import Row, RowItem, orjson_dumps
from pyarrow.parquet import ParquetFile

from worker.dtos import RowsContent

MAX_IMAGE_PIXELS = 10_000_000_000
# ^ see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.MAX_IMAGE_PIXELS


def get_json_size(obj: Any) -> int:
    """Returns the size of an object in bytes once serialized as JSON

    Args:
        obj (Any): the Python object

    Returns:
        int: the size of the serialized object in bytes
    """
    return len(orjson_dumps(obj))


# from https://stackoverflow.com/a/43848928/7351594
def utf8_lead_byte(b: int) -> bool:
    """A UTF-8 intermediate byte starts with the bits 10xxxxxx."""
    return (b & 0xC0) != 0x80


def utf8_byte_truncate(text: str, max_bytes: int) -> str:
    """If text[max_bytes] is not a lead byte, back up until a lead byte is
    found and truncate before that character."""
    utf8 = text.encode("utf8")
    if len(utf8) <= max_bytes:
        return text
    i = max_bytes
    while i > 0 and not utf8_lead_byte(utf8[i]):
        i -= 1
    return utf8[:i].decode("utf8", "ignore")


# Mutates row_item, and returns it anyway
def truncate_row_item(row_item: RowItem, min_cell_bytes: int, columns_to_keep_untruncated: List[str]) -> RowItem:
    row = {}
    for column_name, cell in row_item["row"].items():
        # for now: all the cells above min_cell_bytes are truncated to min_cell_bytes
        # it's done by replacing the cell (which can have any type) by a string with
        # its JSON serialization, and then truncating it to min_cell_bytes
        cell_json = orjson_dumps(cell)
        if len(cell_json) <= min_cell_bytes or column_name in columns_to_keep_untruncated:
            row[column_name] = cell
        else:
            cell_json_str = cell_json.decode("utf8", "ignore")
            row_item["truncated_cells"].append(column_name)
            row[column_name] = utf8_byte_truncate(text=cell_json_str, max_bytes=min_cell_bytes)
    row_item["row"] = row
    # row_idx = row_item["row_idx"]
    # logging.debug(f"the size of the rows is now ({rows_bytes}) after truncating row idx={row_idx}")
    return row_item


COMMA_SIZE = 1  # the comma "," is encoded with one byte in utf-8


# Mutates row_items, and returns them anyway
def truncate_row_items(
    row_items: List[RowItem], min_cell_bytes: int, rows_max_bytes: int, columns_to_keep_untruncated: List[str]
) -> List[RowItem]:
    # compute the current size
    rows_bytes = sum(get_json_size(row_item) for row_item in row_items) + COMMA_SIZE * (len(row_items) - 1)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_json_size(row_item) + COMMA_SIZE
        row_item = truncate_row_item(
            row_item=row_item, min_cell_bytes=min_cell_bytes, columns_to_keep_untruncated=columns_to_keep_untruncated
        )
        new_size = get_json_size(row_item) + COMMA_SIZE
        rows_bytes += new_size - previous_size
    return row_items


def to_row_item(row_idx: int, row: Row) -> RowItem:
    return {
        "row_idx": row_idx,
        "row": row,
        "truncated_cells": [],
    }


def create_truncated_row_items(
    rows: List[Row],
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_min_number: int,
    columns_to_keep_untruncated: List[str],
) -> List[RowItem]:
    row_items = []
    rows_bytes = 0

    # two restrictions must be enforced:
    # - at least rows_min_number rows
    # - at most rows_max_bytes bytes. Note that it's the limit to the sum of the rows sizes. The JSON response size
    #   will be greater, due to the other fields (row_idx, truncated_cells, features, etc.).
    # To enforce this:
    # 1. first get the first rows_min_number rows
    for row_idx, row in enumerate(rows[:rows_min_number]):
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        row_items.append(row_item)

    # 2. if the total is over the bytes limit, truncate the values, iterating backwards starting
    # from the last rows, until getting under the threshold
    # caveat: the truncation might not be enough to get under the threshold if:
    # - the number of columns is too high
    # - rows_max_bytes is too low (or even negative)
    if rows_bytes >= rows_max_bytes:
        # logging.debug(
        #     f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
        #     f" ({rows_max_bytes}), they will be truncated"
        # )
        return truncate_row_items(
            row_items=row_items,
            min_cell_bytes=min_cell_bytes,
            rows_max_bytes=rows_max_bytes,
            columns_to_keep_untruncated=columns_to_keep_untruncated,
        )

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_idx = rows_min_number + idx
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        if rows_bytes >= rows_max_bytes:
            # logging.debug(
            #     f"the rows in the split have been truncated to {row_idx} row(s) to keep the size"
            #     f" ({rows_bytes}) under the limit ({rows_max_bytes})"
            # )
            break
        row_items.append(row_item)
    return row_items


FuncT = TypeVar("FuncT", bound=Callable[..., Any])
RETRY_SLEEPS = (1, 1, 1, 10, 10, 10, 60, 60, 60, 10 * 60)
RETRY_ON: Tuple[Type[Exception]] = (Exception,)


class retry:
    """retries with an increasing sleep before every attempt"""

    def __init__(self, sleeps: Sequence[int] = RETRY_SLEEPS, on: Sequence[Type[Exception]] = RETRY_ON) -> None:
        self.sleeps = sleeps
        self.on = on

    def __call__(self, func: FuncT) -> FuncT:
        @functools.wraps(func)
        def decorator(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            last_err = None
            while attempt < len(self.sleeps):
                try:
                    """always sleep before calling the function. It will prevent rate limiting in the first place"""
                    duration = self.sleeps[attempt]
                    logging.info(f"Sleep during {duration} seconds to preventively mitigate rate limiting.")
                    time.sleep(duration)
                    return func(*args, **kwargs)
                except tuple(self.on) as err:
                    logging.info(f"Got a {type(err)}. Let's retry.")
                    last_err = err
                    attempt += 1
            raise RuntimeError(f"Give up after {attempt} attempts. The last one raised {type(last_err)}") from last_err

        return cast(FuncT, decorator)


@retry(on=[ConnectionError])
def get_rows(
    dataset: str,
    config: str,
    split: str,
    streaming: bool,
    rows_max_number: int,
    token: Union[bool, str, None] = False,
    column_names: Optional[List[str]] = None,
) -> RowsContent:
    download_config = DownloadConfig(delete_extracted=True, token=token)
    PIL.Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    ds = load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=streaming,
        token=token,
        download_config=download_config,
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
    return RowsContent(rows=rows, all_fetched=all_fetched)


def get_rows_or_raise(
    dataset: str,
    config: str,
    split: str,
    rows_max_number: int,
    token: Union[bool, str, None],
    info: DatasetInfo,
    max_size_fallback: Optional[int] = None,
    column_names: Optional[List[str]] = [],
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
        )
    except Exception as err:
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

HF_HUB_HTTP_ERROR_RETRY_SLEEPS = [1, 1, 1, 10, 10, 10]
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


def check_split_exists(dataset: str, config: str, split: str) -> None:
    """
    Check if dataset has a provided split in a provided config. Dataset's splits are taken from the best response
    of 'config-split-names-from-streaming' and 'config-split-names-from-info' steps' cache.
    """
    split_names_best_response = get_previous_step_or_raise(
        kinds=["config-split-names-from-streaming", "config-split-names-from-info"], dataset=dataset, config=config
    )
    try:
        splits_content = split_names_best_response.response["content"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError(
            (
                "Previous steps 'config-split-names-from-streaming' and 'config-split-names-from-info did not return"
                " the expected content."
            ),
            e,
        ) from e

    if split not in [split_item["split"] for split_item in splits_content]:
        raise SplitNotFoundError(f"Split '{split}' does not exist for the config '{config}' of the dataset.")
