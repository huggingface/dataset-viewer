# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from functools import partial
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional

import pyarrow as pa
from datasets import Features
from fsspec import AbstractFileSystem  # type: ignore
from hffs.fs import HfFileSystem
from libcommon.constants import (
    PARQUET_REVISION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import SplitFullName
from libcommon.storage import StrPath
from libcommon.viewer_utils.features import get_cell_value
from pyarrow.parquet import ParquetFile
from tqdm.contrib.concurrent import thread_map

from worker.config import AppConfig, FirstRowsConfig
from worker.job_runner import (
    CompleteJobResult,
    JobRunner,
    JobRunnerError,
    get_previous_step_or_raise,
)
from worker.utils import (
    Row,
    RowItem,
    SplitFirstRowsResponse,
    create_truncated_row_items,
    get_json_size,
    to_features_list,
)

SplitFirstRowsFromParquetJobRunnerErrorCode = Literal[
    "RowsPostProcessingError",
    "TooManyColumnsError",
    "TooBigContentError",
    "PreviousStepFormatError",
    "ParquetResponseEmptyError",
    "FileSystemError",
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


class PreviousStepFormatError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class ParquetResponseEmptyError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when no parquet files were found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ParquetResponseEmptyError", cause, False)


class FileSystemError(SplitFirstRowsFromParquetJobRunnerError):
    """Raised when an error happen reading from File System."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FileSystemError", cause, False)


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


def get_parquet_fs(dataset: str, hf_token: Optional[str]) -> AbstractFileSystem:
    """Get the parquet filesystem for a dataset.
    The parquet files are stored in a separate branch of the dataset repository (see PARQUET_REVISION)
    Args:
        dataset (str): The dataset name.
        hf_token (Optional[str]): The token to access the filesystem.
    Returns:
        HfFileSystem: The parquet filesystem.
    """
    return HfFileSystem(dataset, repo_type="dataset", revision=PARQUET_REVISION, token=hf_token)


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

    config_parquet_best_response = get_previous_step_or_raise(kinds=["config-parquet"], dataset=dataset, config=config)
    try:
        parquet_files_content = config_parquet_best_response.response["content"]["parquet_files"]
        sources = sorted(
            f"{config}/{parquet_file['filename']}"
            for parquet_file in parquet_files_content
            if parquet_file["split"] == split and parquet_file["config"] == config
        )
        if not sources:
            raise ParquetResponseEmptyError("No parquet files found.")
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
    num_rows = 0
    last_row_group_id = 0
    row_group_readers = []
    for parquet_file in parquet_files:
        for group_id in range(parquet_file.metadata.num_row_groups):
            last_row_group_id = group_id
            row_group_readers.append(partial(parquet_file.read_row_group, i=group_id))
            if num_rows + parquet_file.metadata.row_group(group_id).num_rows >= rows_max_number:
                num_rows = rows_max_number
                break
            else:
                num_rows += parquet_file.metadata.row_group(group_id).num_rows
        else:
            continue
        break

    if len(row_group_readers) == 0:
        raise ParquetResponseEmptyError("No parquet files found.")

    pa_table = pa.concat_tables([row_group_readers[i]() for i in range(0, last_row_group_id + 1)])
    result = pa_table.slice(0, num_rows)

    rows = [
        RowItem(
            {
                "row_idx": idx,
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
    return response


class SplitFirstRowsFromParquetJobRunner(JobRunner):
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
        assets_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            common_config=app_config.common,
            worker_config=app_config.worker,
            processing_step=processing_step,
        )
        self.first_rows_config = app_config.first_rows
        self.assets_directory = assets_directory
        self.assets_base_url = app_config.assets.base_url

    def compute(self) -> CompleteJobResult:
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        self.raise_if_parallel_response_exists(
            parallel_cache_kind="split-first-rows-from-streaming",
            parallel_job_version=PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
        )
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
