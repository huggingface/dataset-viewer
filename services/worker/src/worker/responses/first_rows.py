# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import itertools
import logging
import sys
from typing import Any, Dict, List, Optional, TypedDict

from datasets import (
    Dataset,
    Features,
    IterableDataset,
    get_dataset_config_info,
    load_dataset,
)
from libutils.utils import orjson_dumps

from worker.config import MIN_CELL_BYTES
from worker.constants import DEFAULT_ROWS_MAX_BYTES, DEFAULT_ROWS_MAX_NUMBER
from worker.features import get_cell_value
from worker.responses.splits import get_splits_response
from worker.utils import (
    ConfigNotFoundError,
    FeaturesError,
    InfoError,
    NormalRowsError,
    RowsPostProcessingError,
    SplitNotFoundError,
    StreamingRowsError,
    retry,
)

logger = logging.getLogger(__name__)


Row = Dict[str, Any]


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Dict[str, Any]


class RowItem(TypedDict):
    row_idx: int
    row: Dict[str, Any]
    truncated_cells: List[str]


class FirstRowsResponse(TypedDict):
    dataset: str
    config: str
    split: str
    features: List[FeatureItem]
    rows: List[RowItem]


@retry(logger=logger)
def get_rows(
    dataset: str,
    config: str,
    split: str,
    streaming: bool,
    rows_max_number: int,
    hf_token: Optional[str] = None,
) -> List[Row]:
    ds = load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=streaming,
        use_auth_token=hf_token,
    )
    if streaming:
        if not isinstance(ds, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset in streaming mode")
    elif not isinstance(ds, Dataset):
        raise TypeError("load_dataset should return a Dataset in normal mode")
    rows_plus_one = list(itertools.islice(ds, rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
    if len(rows_plus_one) <= rows_max_number:
        logger.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logger.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return rows_plus_one[:rows_max_number]


def get_size_in_bytes(obj: Any):
    return sys.getsizeof(orjson_dumps(obj))
    # ^^ every row is transformed here in a string, because it corresponds to
    # the size the row will contribute in the JSON response to /rows endpoint.
    # The size of the string is measured in bytes.
    # An alternative would have been to look at the memory consumption (pympler) but it's
    # less related to what matters here (size of the JSON, number of characters in the
    # dataset viewer table on the hub)


def truncate_cell(cell: Any, min_cell_bytes: int) -> str:
    return orjson_dumps(cell)[:min_cell_bytes].decode("utf8", "ignore")


# Mutates row_item, and returns it anyway
def truncate_row_item(row_item: RowItem) -> RowItem:
    row = {}
    for column_name, cell in row_item["row"].items():
        # for now: all the cells, but the smallest ones, are truncated
        cell_bytes = get_size_in_bytes(cell)
        if cell_bytes > MIN_CELL_BYTES:
            row_item["truncated_cells"].append(column_name)
            row[column_name] = truncate_cell(cell, MIN_CELL_BYTES)
        else:
            row[column_name] = cell
    row_item["row"] = row
    return row_item


# Mutates row_items, and returns them anyway
def truncate_row_items(row_items: List[RowItem], rows_max_bytes: int) -> List[RowItem]:
    # compute the current size
    rows_bytes = sum(get_size_in_bytes(row_item) for row_item in row_items)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_size_in_bytes(row_item)
        row_item = truncate_row_item(row_item)
        new_size = get_size_in_bytes(row_item)
        rows_bytes += new_size - previous_size
        row_idx = row_item["row_idx"]
        logger.debug(f"the size of the rows is now ({rows_bytes}) after truncating row idx={row_idx}")
    return row_items


def to_row_item(dataset: str, config: str, split: str, row_idx: int, row: Row) -> RowItem:
    return {
        "row_idx": row_idx,
        "row": row,
        "truncated_cells": [],
    }


def create_truncated_row_items(
    dataset: str,
    config: str,
    split: str,
    rows: List[Row],
    rows_max_bytes: int,
    rows_min_number: int,
) -> List[RowItem]:
    row_items = []
    rows_bytes = 0

    # two restrictions must be enforced:
    # - at least rows_min_number rows
    # - at most rows_max_bytes bytes
    # To enforce this:
    # 1. first get the first rows_min_number rows
    for row_idx, row in enumerate(rows[:rows_min_number]):
        row_item = to_row_item(dataset, config, split, row_idx, row)
        rows_bytes += get_size_in_bytes(row_item)
        row_items.append(row_item)

    # 2. if the total is over the bytes limit, truncate the values, iterating backwards starting
    # from the last rows, until getting under the threshold
    if rows_bytes >= rows_max_bytes:
        logger.debug(
            f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
            f" ({rows_max_bytes}), they will be truncated"
        )
        return truncate_row_items(row_items, rows_max_bytes)

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_idx = rows_min_number + idx
        row_item = to_row_item(dataset, config, split, row_idx, row)
        rows_bytes += get_size_in_bytes(row_item)
        if rows_bytes >= rows_max_bytes:
            logger.debug(
                f"the rows in the split have been truncated to {row_idx} row(s) to keep the size"
                f" ({rows_bytes}) under the limit ({rows_max_bytes})"
            )
            break
        row_items.append(row_item)
    return row_items


def transform_rows(
    dataset: str, config: str, split: str, rows: List[Row], features: Features, assets_base_url: str
) -> List[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset,
                config,
                split,
                row_idx,
                row[featureName],
                featureName,
                fieldType,
                assets_base_url,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


# in JSON, dicts do not carry any order, so we need to return a list
#
# > An object is an *unordered* collection of zero or more name/value pairs, where a name is a string and a value
#   is a string, number, boolean, null, object, or array.
# > An array is an *ordered* sequence of zero or more values.
# > The terms "object" and "array" come from the conventions of JavaScript.
# from https://stackoverflow.com/a/7214312/7351594 / https://www.rfc-editor.org/rfc/rfc7159.html
def to_features_list(dataset: str, config: str, split: str, features: Features) -> List[FeatureItem]:
    features_dict = features.to_dict()
    return [
        {
            "feature_idx": idx,
            "name": name,
            "type": features_dict[name],
        }
        for idx, name in enumerate(features)
    ]


def get_first_rows_response(
    dataset: str,
    config: str,
    split: str,
    assets_base_url: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> FirstRowsResponse:
    """
    Get the response of /first-rows for one specific split of a dataset from huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        assets_base_url (`str`):
            The base url of the assets.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        max_size_fallback (`int`, *optional*):
            The maximum number of bytes of the split to fallback to normal mode if the streaming mode fails. If None,
            it will not fallback to normal mode. Defaults to None.
        rows_max_bytes (`int`, *optional*):
            The maximum number of bytes of the response (else, the response is truncated). Defaults to 1_000_000 bytes.
        rows_max_number (`int`, *optional*):
            The maximum number of rows of the response. Defaults to 100.
        rows_min_number (`int`, *optional*):
            The minimum number of rows of the response. Defaults to 0.
    Returns:
        [`FirstRowsResponse`]: The list of first rows of the split.
    <Tip>
    Raises the following errors:
        - [`~worker.exceptions.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~worker.exceptions.ConfigNotFoundError`]
          If the config does not exist in the dataset.
        - [`~worker.exceptions.SplitNotFoundError`]
          If the split does not exist in the dataset.
        - [`~worker.utils.InfoError`]
          If the config info could not be obtained using the datasets library.
        - [`~worker.utils.FeaturesError`]
          If the split features could not be obtained using the datasets library.
        - [`~worker.utils.StreamingRowsError`]
          If the split rows could not be obtained using the datasets library in streaming mode.
        - [`~worker.utils.NormalRowsError`]
          If the split rows could not be obtained using the datasets library in normal mode.
        - [`~worker.utils.RowsPostProcessingError`]
          If the post-processing of the split rows failed, e.g. while saving the images or audio files to the assets.
    </Tip>
    """
    logger.info(f"get first-rows for dataset={dataset} config={config} split={split}")
    if rows_max_bytes is None:
        rows_max_bytes = DEFAULT_ROWS_MAX_BYTES
    if rows_max_number is None:
        rows_max_number = DEFAULT_ROWS_MAX_NUMBER
    if rows_min_number is None:
        rows_min_number = 0
    # first ensure the tuple (dataset, config, split) exists on the Hub
    splits_response = get_splits_response(dataset, hf_endpoint, hf_token)
    # ^ can raise DatasetNotFoundError or SplitsNamesError
    if config not in [split_item["config"] for split_item in splits_response["splits"]]:
        raise ConfigNotFoundError(f"config {config} does not exist for dataset {dataset}")
    if {"dataset": dataset, "config": config, "split": split} not in [
        {
            "dataset": split_item["dataset"],
            "config": split_item["config"],
            "split": split_item["split"],
        }
        for split_item in splits_response["splits"]
    ]:
        raise SplitNotFoundError("The config or the split does not exist in the dataset")
    # get the features
    try:
        info = get_dataset_config_info(
            path=dataset,
            config_name=config,
            use_auth_token=hf_token,
        )
    except Exception as err:
        raise InfoError("The info cannot be fetched for the dataset config.", cause=err) from err
    if not info.features:
        try:
            # https://github.com/huggingface/datasets/blob/f5826eff9b06ab10dba1adfa52543341ef1e6009/src/datasets/iterable_dataset.py#L1255
            iterable_dataset = load_dataset(
                dataset,
                name=config,
                split=split,
                streaming=True,
                use_auth_token=hf_token,
            )
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset")
            iterable_dataset = iterable_dataset._resolve_features()
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset")
            features = iterable_dataset.features
        except Exception as err:
            raise FeaturesError("The split features (columns) cannot be extracted.", cause=err) from err
    else:
        features = info.features
    # get the rows
    try:
        rows = get_rows(dataset, config, split, streaming=True, rows_max_number=rows_max_number, hf_token=hf_token)
    except Exception as err:
        if max_size_fallback is None or info.size_in_bytes is None or info.size_in_bytes > max_size_fallback:
            raise StreamingRowsError(
                "Cannot load the dataset split (in streaming mode) to extract the first rows.",
                cause=err,
            ) from err
        try:
            rows = get_rows(
                dataset,
                config,
                split,
                streaming=False,
                rows_max_number=rows_max_number,
                hf_token=hf_token,
            )
        except Exception as err:
            raise NormalRowsError(
                "Cannot load the dataset split (in normal download mode) to extract the first rows.",
                cause=err,
            ) from err
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(dataset, config, split, rows, features, assets_base_url)
    except Exception as err:
        raise RowsPostProcessingError(
            "Server error while post-processing the split rows. Please report the issue.",
            cause=err,
        ) from err
    # truncate the rows to fit within the restrictions, and prepare them as RowItems
    row_items = create_truncated_row_items(dataset, config, split, transformed_rows, rows_max_bytes, rows_min_number)
    # return the response
    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": to_features_list(dataset, config, split, features),
        "rows": row_items,
    }
