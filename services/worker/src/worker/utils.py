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
    Mapping,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from datasets import (
    Dataset,
    DatasetInfo,
    DownloadConfig,
    Features,
    IterableDataset,
    load_dataset,
)
from libcommon.utils import orjson_dumps

from worker.common_exceptions import NormalRowsError, StreamingRowsError


class DatasetItem(TypedDict):
    dataset: str


class ConfigItem(DatasetItem):
    config: Optional[str]


class SplitItem(ConfigItem):
    split: Optional[str]


class SplitsList(TypedDict):
    splits: List[SplitItem]


class FailedConfigItem(ConfigItem):
    error: Mapping[str, Any]


class DatasetSplitNamesResponse(TypedDict):
    splits: List[SplitItem]
    pending: List[ConfigItem]
    failed: List[FailedConfigItem]


class PreviousJob(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Mapping[str, Any]


class RowItem(TypedDict):
    row_idx: int
    row: Mapping[str, Any]
    truncated_cells: List[str]


class SplitFirstRowsResponse(TypedDict):
    dataset: str
    config: str
    split: str
    features: List[FeatureItem]
    rows: List[RowItem]


class OptUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


class OptInOutUrlsScanResponse(TypedDict):
    urls_columns: List[str]
    num_opt_in_urls: int
    num_opt_out_urls: int
    num_urls: int
    num_scanned_rows: int
    has_urls_columns: bool


class OptInOutUrlsScanDetailedResponse(OptInOutUrlsScanResponse):
    opt_in_urls: List[OptUrl]
    opt_out_urls: List[OptUrl]


# in JSON, dicts do not carry any order, so we need to return a list
#
# > An object is an *unordered* collection of zero or more name/value pairs, where a name is a string and a value
#   is a string, number, boolean, null, object, or array.
# > An array is an *ordered* sequence of zero or more values.
# > The terms "object" and "array" come from the conventions of JavaScript.
# from https://stackoverflow.com/a/7214312/7351594 / https://www.rfc-editor.org/rfc/rfc7159.html
def to_features_list(features: Features) -> List[FeatureItem]:
    features_dict = features.to_dict()
    return [
        {
            "feature_idx": idx,
            "name": name,
            "type": features_dict[name],
        }
        for idx, name in enumerate(features)
    ]


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
def truncate_row_item(row_item: RowItem, min_cell_bytes: int) -> RowItem:
    row = {}
    for column_name, cell in row_item["row"].items():
        # for now: all the cells above min_cell_bytes are truncated to min_cell_bytes
        # it's done by replacing the cell (which can have any type) by a string with
        # its JSON serialization, and then truncating it to min_cell_bytes
        cell_json = orjson_dumps(cell)
        if len(cell_json) <= min_cell_bytes:
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
def truncate_row_items(row_items: List[RowItem], min_cell_bytes: int, rows_max_bytes: int) -> List[RowItem]:
    # compute the current size
    rows_bytes = sum(get_json_size(row_item) for row_item in row_items) + COMMA_SIZE * (len(row_items) - 1)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_json_size(row_item) + COMMA_SIZE
        row_item = truncate_row_item(row_item=row_item, min_cell_bytes=min_cell_bytes)
        new_size = get_json_size(row_item) + COMMA_SIZE
        rows_bytes += new_size - previous_size
    return row_items


Row = Mapping[str, Any]


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
        return truncate_row_items(row_items=row_items, min_cell_bytes=min_cell_bytes, rows_max_bytes=rows_max_bytes)

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


def retry(func: FuncT) -> FuncT:
    """retries with an increasing sleep before every attempt"""
    SLEEPS = [1, 7, 70, 7 * 60, 70 * 60]
    MAX_ATTEMPTS = len(SLEEPS)

    @functools.wraps(func)
    def decorator(*args: Any, **kwargs: Any) -> Any:
        attempt = 0
        last_err = None
        while attempt < MAX_ATTEMPTS:
            try:
                """always sleep before calling the function. It will prevent rate limiting in the first place"""
                duration = SLEEPS[attempt]
                logging.info(f"Sleep during {duration} seconds to preventively mitigate rate limiting.")
                time.sleep(duration)
                return func(*args, **kwargs)
            except ConnectionError as err:
                logging.info("Got a ConnectionError, possibly due to rate limiting. Let's retry.")
                last_err = err
                attempt += 1
        raise RuntimeError(f"Give up after {attempt} attempts with ConnectionError") from last_err

    return cast(FuncT, decorator)


@retry
def get_rows(
    dataset: str,
    config: str,
    split: str,
    streaming: bool,
    rows_max_number: int,
    use_auth_token: Union[bool, str, None] = False,
    column_names: Optional[List[str]] = None,
) -> List[Row]:
    download_config = DownloadConfig(delete_extracted=True)
    ds = load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=streaming,
        use_auth_token=use_auth_token,
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
    if len(rows_plus_one) <= rows_max_number:
        logging.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logging.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return rows_plus_one[:rows_max_number]


def get_rows_or_raise(
    dataset: str,
    config: str,
    split: str,
    rows_max_number: int,
    use_auth_token: Union[bool, str, None],
    info: DatasetInfo,
    max_size_fallback: Optional[int] = None,
    column_names: Optional[List[str]] = [],
) -> List[Row]:
    try:
        return get_rows(
            dataset=dataset,
            config=config,
            split=split,
            streaming=True,
            rows_max_number=rows_max_number,
            use_auth_token=use_auth_token,
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
                use_auth_token=use_auth_token,
            )
        except Exception as err:
            raise NormalRowsError(
                "Cannot load the dataset split (in normal download mode) to extract the first rows.",
                cause=err,
            ) from err
