# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import functools
import logging
import time
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, List, Literal, Mapping, Optional, TypeVar, cast

import numpy as np
import pyarrow as pa
from datasets import Features
from hffs.fs import HfFileSystem
from libcommon.constants import (
    PARQUET_REVISION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response
from libcommon.storage import StrPath
from libcommon.utils import orjson_dumps
from pyarrow.parquet import ParquetFile
from tqdm.contrib.concurrent import thread_map

from worker.config import AppConfig, FirstRowsConfig
from worker.features import get_cell_value
from worker.job_runner import CompleteJobResult, ConfigNotFoundError, JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.utils import (
    RowItem,
    SplitFirstRowsResponse,
    get_json_size,
    to_features_list,
)

SplitFirstRowsFromParquetJobRunnerErrorCode = Literal[
    "SplitsNamesError",
    "EmptyDatasetError",
    "InfoError",
    "FeaturesError",
    "StreamingRowsError",
    "NormalRowsError",
    "RowsPostProcessingError",
    "TooManyColumnsError",
    "TooBigContentError",
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "ParquetResponseEmptyError",
]


class SplitFirstRowsFromParquetJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SplitFirstRowsFromParquetJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class SplitsNamesError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class EmptyDatasetError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class InfoError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class FeaturesError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the features could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesError", cause, True)


class StreamingRowsError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class NormalRowsError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)


class RowsPostProcessingError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the rows could not be post-processed successfully."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "RowsPostProcessingError", cause, False)


class TooManyColumnsError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class TooBigContentError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the first rows content exceeded the max size of bytes."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooBigContentError", cause, False)


class PreviousStepStatusError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class ParquetResponseEmptyError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when no parquet files were found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ParquetResponseEmptyError", cause, False)


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


Row = Mapping[str, Any]


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
    rows: List[RowItem],
    features: Features,
    assets_base_url: str,
    assets_directory: StrPath,
) -> List[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=row["row"][featureName] if featureName in row["row"] else None,
                featureName=featureName,
                fieldType=fieldType,
                assets_base_url=assets_base_url,
                assets_directory=assets_directory,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


def get_parquet_fs(dataset: str, hf_token: Optional[str]) -> HfFileSystem:
    """Get the parquet filesystem for a dataset.
    The parquet files are stored in a separate branch of the dataset repository (see PARQUET_REVISION)
    Args:
        dataset (str): The dataset name.
        hf_token (Optional[str]): The token to access the filesystem.
    Returns:
        HfFileSystem: The parquet filesystem.
    """
    return HfFileSystem(dataset, repo_type="dataset", revision=PARQUET_REVISION, token=hf_token)


class FileSystemError(Exception):
    pass


def compute_first_rows_response(
    dataset: str,
    config: str,
    split: str,
    assets_base_url: str,
    hf_token: Optional[str],
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    assets_directory: StrPath,
) -> SplitFirstRowsResponse:
    logging.info(f"get first-rows for dataset={dataset} config={config} split={split}")

    # first ensure the tuple (dataset, config, split) exists on the Hub
    try:
        upstream_response = get_response(kind="config-parquet", dataset=dataset, config=config)
        if upstream_response["http_status"] != HTTPStatus.OK:
            raise PreviousStepStatusError(
                f"Previous step gave an error: {upstream_response['http_status']}. This job should not have been"
                " created."
            )

        parquet_files_content = upstream_response["content"]["parquet_files"]
        sources = sorted(
            f"{config}/{parquet_file['filename']}"
            for parquet_file in parquet_files_content
            if parquet_file["split"] == split and parquet_file["config"] == config
        )
        if not sources:
            raise ParquetResponseEmptyError("No parquet files found.")
    except DoesNotExist:
        raise ConfigNotFoundError(f"The config '{config}' does not exist for the dataset.'")
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    logging.debug(f"Found {len(sources)} parquet files for {dataset=}, {config=}, {split=}: {sources}")

    fs = get_parquet_fs(dataset=dataset, hf_token=hf_token)
    desc = f"{dataset}/{config}/{split}"
    try:
        parquet_files: List[ParquetFile] = thread_map(
            partial(ParquetFile, filesystem=fs), sources, desc=desc, unit="pq", disable=True
        )
    except Exception as e:
        raise FileSystemError(f"Could not read the parquet files: {e}") from e

    # get the features
    features = Features.from_arrow_schema(parquet_files[0].schema.to_arrow_schema())

    if features and len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(features)}) exceeds the maximum supported number of columns"
            f" ({columns_max_number}). This is a current limitation of the datasets viewer. You can reduce the number"
            " of columns if you want the viewer to work."
        )

    # validate size of response without the rows
    features_list = to_features_list(features=features)
    response_features_only: SplitFirstRowsResponse = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": features_list,
        "rows": [],
    }

    surrounding_json_size = get_json_size(response_features_only)
    if surrounding_json_size > rows_max_bytes:
        raise TooBigContentError(
            f"The size of the content of the first rows ({surrounding_json_size}) exceeds the maximum"
            f" supported size ({rows_max_bytes} B) even after truncation. Please report the issue."
        )

    # get the rows
    # TODO: Not sure if it is needed to get all the offsets yet
    row_group_offsets = np.cumsum(
        [
            parquet_file.metadata.row_group(group_id).num_rows
            for parquet_file in parquet_files
            for group_id in range(parquet_file.metadata.num_row_groups)
        ]
    )
    row_group_readers = [
        partial(parquet_file.read_row_group, i=group_id)
        for parquet_file in parquet_files
        for group_id in range(parquet_file.metadata.num_row_groups)
    ]

    if (len(row_group_offsets) == 0) or (len(row_group_readers) == 0):
        raise ParquetResponseEmptyError("No parquet files found.")

    # TODO: Remove unnecessary code here, only first N rows is needed
    offset = 0
    length = rows_max_number
    last_row_in_parquet = row_group_offsets[-1] - 1
    first_row = min(offset, last_row_in_parquet)
    last_row = min(offset, offset + length - 1, last_row_in_parquet)
    first_row_group_id, last_row_group_id = np.searchsorted(row_group_offsets, [first_row, last_row], side="right")
    pa_table = pa.concat_tables([row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)])
    result = pa_table.slice(offset, length)

    rows = [
        RowItem(
            {
                "row_idx": idx + offset,
                "row": row,
                "truncated_cells": [],
            }
        )
        for idx, row in enumerate(result.to_pylist())
    ]

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
    row_items = create_truncated_row_items(
        rows=transformed_rows,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes - surrounding_json_size,
        rows_min_number=rows_min_number,
    )

    response = response_features_only
    response["rows"] = row_items

    # return the response
    return response


class SplitFirstRowsFromParquetJobRunner(DatasetsBasedJobRunner):
    assets_directory: StrPath
    first_rows_config: FirstRowsConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-first-rows-from-parquet"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        first_rows_config: FirstRowsConfig,
        hf_datasets_cache: Path,
        assets_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.first_rows_config = first_rows_config
        self.assets_directory = assets_directory
        self.assets_base_url = app_config.assets.base_url

    def compute(self) -> CompleteJobResult:
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return CompleteJobResult(
            compute_first_rows_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                assets_base_url=self.assets_base_url,
                assets_directory=self.assets_directory,
                hf_token=self.common_config.hf_token,
                min_cell_bytes=self.first_rows_config.min_cell_bytes,
                rows_max_bytes=self.first_rows_config.max_bytes,
                rows_max_number=self.first_rows_config.max_number,
                rows_min_number=self.first_rows_config.min_number,
                columns_max_number=self.first_rows_config.columns_max_number,
            )
        )

    def get_new_splits(self, _: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
