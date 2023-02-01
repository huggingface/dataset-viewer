# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import functools
import itertools
import logging
import time
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict, Union

from datasets import (
    Dataset,
    Features,
    IterableDataset,
    get_dataset_config_info,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.exceptions import CustomError
from libcommon.simple_cache import SplitFullName as _SplitFullName
from libcommon.utils import orjson_dumps

from datasets_based.config import AppConfig, FirstRowsConfig
from datasets_based.features import get_cell_value
from datasets_based.worker import ConfigNotFoundError, JobInfo, SplitNotFoundError
from datasets_based.workers._datasets_based_worker import DatasetsBasedWorker

FirstRowsWorkerErrorCode = Literal[
    "SplitsNamesError",
    "EmptyDatasetError",
    "InfoError",
    "FeaturesError",
    "StreamingRowsError",
    "NormalRowsError",
    "RowsPostProcessingError",
    "TooManyColumnsError",
    "TooBigContentError",
]


class FirstRowsWorkerError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: FirstRowsWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class SplitsNamesError(FirstRowsWorkerError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class EmptyDatasetError(FirstRowsWorkerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class InfoError(FirstRowsWorkerError):
    """Raised when the info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class FeaturesError(FirstRowsWorkerError):
    """Raised when the features could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesError", cause, True)


class StreamingRowsError(FirstRowsWorkerError):
    """Raised when the rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class NormalRowsError(FirstRowsWorkerError):
    """Raised when the rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)


class RowsPostProcessingError(FirstRowsWorkerError):
    """Raised when the rows could not be post-processed successfully."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "RowsPostProcessingError", cause, False)


class TooManyColumnsError(FirstRowsWorkerError):
    """Raised when the dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class TooBigContentError(FirstRowsWorkerError):
    """Raised when the first rows content exceeded the max size of bytes."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooBigContentError", cause, True)


def retry():
    def decorator_retry(func):
        """retries with an increasing sleep before every attempt"""
        SLEEPS = [1, 7, 70, 7 * 60, 70 * 60]
        MAX_ATTEMPTS = len(SLEEPS)

        @functools.wraps(func)
        def decorator(*args, **kwargs):
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

        return decorator

    return decorator_retry


Row = Mapping[str, Any]


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Mapping[str, Any]


class RowItem(TypedDict):
    row_idx: int
    row: Mapping[str, Any]
    truncated_cells: List[str]


class FirstRowsResponse(TypedDict):
    dataset: str
    config: str
    split: str
    features: List[FeatureItem]
    rows: List[RowItem]


@retry()
def get_rows(
    dataset: str,
    config: str,
    split: str,
    streaming: bool,
    rows_max_number: int,
    use_auth_token: Union[bool, str, None] = False,
) -> List[Row]:
    ds = load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=streaming,
        use_auth_token=use_auth_token,
    )
    if streaming:
        if not isinstance(ds, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset in streaming mode")
    elif not isinstance(ds, Dataset):
        raise TypeError("load_dataset should return a Dataset in normal mode")
    rows_plus_one = list(itertools.islice(ds, rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
    if len(rows_plus_one) <= rows_max_number:
        logging.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logging.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return rows_plus_one[:rows_max_number]


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
        row_idx = row_item["row_idx"]
        logging.debug(f"the size of the rows is now ({rows_bytes}) after truncating row idx={row_idx}")
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
        logging.debug(
            f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
            f" ({rows_max_bytes}), they will be truncated"
        )
        return truncate_row_items(row_items=row_items, min_cell_bytes=min_cell_bytes, rows_max_bytes=rows_max_bytes)

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_idx = rows_min_number + idx
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        if rows_bytes >= rows_max_bytes:
            logging.debug(
                f"the rows in the split have been truncated to {row_idx} row(s) to keep the size"
                f" ({rows_bytes}) under the limit ({rows_max_bytes})"
            )
            break
        row_items.append(row_item)
    return row_items


def transform_rows(
    dataset: str,
    config: str,
    split: str,
    rows: List[Row],
    features: Features,
    assets_base_url: str,
    assets_directory: str,
) -> List[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=row[featureName] if featureName in row else None,
                featureName=featureName,
                fieldType=fieldType,
                assets_base_url=assets_base_url,
                assets_directory=assets_directory,
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


def validate_content_size(obj: Any, rows_max_bytes: int):
    if get_json_size(obj) > rows_max_bytes:
        raise TooBigContentError("First rows content exceeded max size bytes.")


class SplitFullName(TypedDict):
    dataset: str
    config: str
    split: str


def get_dataset_split_full_names(dataset: str, use_auth_token: Union[bool, str, None] = False) -> List[SplitFullName]:
    logging.info(f"get dataset '{dataset}' split full names")
    return [
        {"dataset": dataset, "config": config, "split": split}
        for config in get_dataset_config_names(path=dataset, use_auth_token=use_auth_token)
        for split in get_dataset_split_names(path=dataset, config_name=config, use_auth_token=use_auth_token)
    ]


def compute_first_rows_response(
    dataset: str,
    config: str,
    split: str,
    assets_base_url: str,
    hf_token: Optional[str],
    min_cell_bytes: int,
    max_size_fallback: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    assets_directory: str,
) -> FirstRowsResponse:
    """
    Get the response of /first-rows for one specific split of a dataset from huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exist and can be accessed using the token.

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
        hf_token (`str` or `None`):
            An authentication token (See https://huggingface.co/settings/token)
        max_size_fallback (`int`):
            The maximum number of bytes of the split to fallback to normal mode if the streaming mode fails.
        rows_max_bytes (`int`):
            The maximum number of bytes of the response (else, the response is truncated).
        rows_max_number (`int`):
            The maximum number of rows of the response.
        rows_min_number (`int`):
            The minimum number of rows of the response.
        columns_max_number (`int`):
            The maximun number of columns allowed.
    Returns:
        [`FirstRowsResponse`]: The list of first rows of the split.
    <Tip>
    Raises the following errors:
        - [`libcommon.worker.ConfigNotFoundError`]
          If the config does not exist in the dataset.
        - [`libcommon.worker.SplitNotFoundError`]
          If the split does not exist in the dataset.
        - [`~workers.first_rows.InfoError`]
          If the config info could not be obtained using the datasets library.
        - [`~workers.first_rows.FeaturesError`]
          If the split features could not be obtained using the datasets library.
        - [`~workers.first_rows.StreamingRowsError`]
          If the split rows could not be obtained using the datasets library in streaming mode.
        - [`~workers.first_rows.NormalRowsError`]
          If the split rows could not be obtained using the datasets library in normal mode.
        - [`~workers.first_rows.RowsPostProcessingError`]
          If the post-processing of the split rows failed, e.g. while saving the images or audio files to the assets.
        - [`~workers.first_rows.TooManyColumnsError`]
          If the response cannot be saved because of too many columns in the dataset.
    </Tip>
    """
    logging.info(f"get first-rows for dataset={dataset} config={config} split={split}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # first ensure the tuple (dataset, config, split) exists on the Hub
    # get the list of splits
    try:
        split_full_names = get_dataset_split_full_names(dataset=dataset, use_auth_token=use_auth_token)
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise SplitsNamesError("Cannot get the split names for the dataset.", cause=err) from err
    # ^ can raise DatasetNotFoundError or SplitsNamesError
    if config not in [split_full_name["config"] for split_full_name in split_full_names]:
        raise ConfigNotFoundError(f"config {config} does not exist for dataset {dataset}")
    if {"dataset": dataset, "config": config, "split": split} not in [
        {
            "dataset": split_full_name["dataset"],
            "config": split_full_name["config"],
            "split": split_full_name["split"],
        }
        for split_full_name in split_full_names
    ]:
        raise SplitNotFoundError("The config or the split does not exist in the dataset")
    # get the features
    try:
        info = get_dataset_config_info(
            path=dataset,
            config_name=config,
            use_auth_token=use_auth_token,
        )
    except Exception as err:
        raise InfoError("The info cannot be fetched for the dataset config.", cause=err) from err
    if not info.features:
        try:
            # https://github.com/huggingface/datasets/blob/f5826eff9b06ab10dba1adfa52543341ef1e6009/src/datasets/iterable_dataset.py#L1255
            iterable_dataset = load_dataset(
                path=dataset,
                name=config,
                split=split,
                streaming=True,
                use_auth_token=use_auth_token,
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

    if len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"Too many columns. The maximum supported number of columns is {columns_max_number}."
        )

    # validate size of response without the rows
    features_list = to_features_list(features=features)
    response: FirstRowsResponse = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": features_list,
        "rows": [],
    }

    validate_content_size(response, rows_max_bytes=rows_max_bytes)

    # get the rows
    try:
        rows = get_rows(
            dataset=dataset,
            config=config,
            split=split,
            streaming=True,
            rows_max_number=rows_max_number,
            use_auth_token=use_auth_token,
        )
    except Exception as err:
        if info.size_in_bytes is None or info.size_in_bytes > max_size_fallback:
            raise StreamingRowsError(
                "Cannot load the dataset split (in streaming mode) to extract the first rows.",
                cause=err,
            ) from err
        try:
            rows = get_rows(
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
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(
            dataset=dataset,
            config=config,
            split=split,
            rows=rows,
            features=features,
            assets_base_url=assets_base_url,
            assets_directory=assets_directory,
        )
    except Exception as err:
        raise RowsPostProcessingError(
            "Server error while post-processing the split rows. Please report the issue.",
            cause=err,
        ) from err

    # truncate the rows to fit within the restrictions, and prepare them as RowItems
    surrounding_json_size = get_json_size(response)
    row_items = create_truncated_row_items(
        rows=transformed_rows,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes - surrounding_json_size,
        rows_min_number=rows_min_number,
    )
    response["rows"] = row_items
    validate_content_size(response, rows_max_bytes=rows_max_bytes)

    # return the response
    return response


class FirstRowsWorker(DatasetsBasedWorker):
    first_rows_config: FirstRowsConfig

    @staticmethod
    def get_job_type() -> str:
        return "/first-rows"

    @staticmethod
    def get_version() -> str:
        return "2.0.0"

    def __init__(self, job_info: JobInfo, app_config: AppConfig, first_rows_config: FirstRowsConfig) -> None:
        super().__init__(job_info=job_info, app_config=app_config)
        self.first_rows_config = first_rows_config

    def compute(self) -> Mapping[str, Any]:
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return compute_first_rows_response(
            dataset=self.dataset,
            config=self.config,
            split=self.split,
            assets_base_url=self.first_rows_config.assets.base_url,
            assets_directory=self.first_rows_config.assets.storage_directory,
            hf_token=self.common_config.hf_token,
            min_cell_bytes=self.first_rows_config.min_cell_bytes,
            max_size_fallback=self.first_rows_config.fallback_max_dataset_size,
            rows_max_bytes=self.first_rows_config.max_bytes,
            rows_max_number=self.first_rows_config.max_number,
            rows_min_number=self.first_rows_config.min_number,
            columns_max_number=self.first_rows_config.columns_max_number,
        )

    def get_new_splits(self, _: Mapping[str, Any]) -> set[_SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {_SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
