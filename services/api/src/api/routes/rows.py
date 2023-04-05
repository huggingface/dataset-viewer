# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import os
import random
import shutil
from functools import lru_cache, partial
from itertools import islice
from os import PathLike
from typing import Any, List, Mapping, Optional, TypedDict, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features
from hffs.fs import HfFileSystem
from libcommon.processing_graph import ProcessingStep
from libcommon.viewer_utils.asset import (
    glob_rows_in_assets_dir,
    update_last_modified_date_of_rows_in_assets_dir,
)
from libcommon.viewer_utils.features import get_cell_value
from starlette.requests import Request
from starlette.responses import Response
from tqdm.contrib.concurrent import thread_map

from api.authentication import auth_check
from api.prometheus import StepProfiler
from api.routes.endpoint import get_cache_entry_from_steps
from api.utils import (
    ApiCustomError,
    Endpoint,
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)

MAX_ROWS = 100

PARQUET_REVISION = "refs/convert/parquet"

CLEAN_CACHE_PROBA = 0.05
KEEP_ROWS_BELOW_INDEX = 100
KEEP_N_MOST_RECENT_ROWS = 1000
MAX_CLEAN_SAMPLE_SIZE = 10_000

StrPath = Union[str, PathLike[str]]


class FileSystemError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class ParquetResponseEmptyError(Exception):
    pass


class ParquetDataProcessingError(Exception):
    pass


# TODO: how to invalidate the cache when the parquet branch is created or deleted?
@lru_cache(maxsize=128)
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


UNSUPPORTED_FEATURES_MAGIC_STRINGS = ["Audio(", "'binary'"]


class RowsIndex:
    def __init__(
        self,
        dataset: str,
        config: str,
        split: str,
        config_parquet_processing_steps: List[ProcessingStep],
        init_processing_steps: List[ProcessingStep],
        hf_endpoint: str,
        hf_token: Optional[str],
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.__post_init__(
            config_parquet_processing_steps=config_parquet_processing_steps,
            init_processing_steps=init_processing_steps,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
        )

    def __post_init__(
        self,
        config_parquet_processing_steps: List[ProcessingStep],
        init_processing_steps: List[ProcessingStep],
        hf_endpoint: str,
        hf_token: Optional[str],
    ) -> None:
        with StepProfiler(method="rows.index", step="all"):
            # get the list of parquet files
            with StepProfiler(method="rows.index", step="get list of parquet files for split"):
                try:
                    result = get_cache_entry_from_steps(
                        processing_steps=config_parquet_processing_steps,
                        dataset=self.dataset,
                        config=self.config,
                        split=None,
                        init_processing_steps=init_processing_steps,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                    )
                    content = result["content"]
                except ApiCustomError as e:
                    raise e
                except Exception as e:
                    raise UnexpectedError("Could not get the list of parquet files to fetch the rows from.") from e
                    # ^ TODO: improve the error, depending on the case
                try:
                    sources = sorted(
                        f"{self.config}/{parquet_file['filename']}"
                        for parquet_file in content["parquet_files"]
                        if parquet_file["split"] == self.split and parquet_file["config"] == self.config
                    )
                except Exception as e:
                    raise ParquetResponseFormatError(f"Could not parse the list of parquet files: {e}") from e
                logging.debug(
                    f"Found {len(sources)} parquet files for dataset={self.dataset}, config={self.config},"
                    f" split={self.split}: {sources}"
                )
                if not sources:
                    raise ParquetResponseEmptyError("No parquet files found.")
            with StepProfiler(method="rows.index", step="get the Hub's dataset filesystem"):
                fs = get_parquet_fs(dataset=self.dataset, hf_token=hf_token)
            with StepProfiler(method="rows.index", step="get one parquet reader per parquet file"):
                desc = f"{self.dataset}/{self.config}/{self.split}"
                try:
                    parquet_files: List[pq.ParquetFile] = thread_map(
                        partial(pq.ParquetFile, filesystem=fs), sources, desc=desc, unit="pq", disable=True
                    )
                except Exception as e:
                    raise FileSystemError(f"Could not read the parquet files: {e}") from e
            with StepProfiler(method="rows.index", step="get the dataset's features"):
                self.features = Features.from_arrow_schema(parquet_files[0].schema.to_arrow_schema())

            self.supported_columns, self.unsupported_columns = [], []
            for column, feature in self.features.items():
                str_feature = str(feature)
                str_column = str(column)
                if any(magic_string in str_feature for magic_string in UNSUPPORTED_FEATURES_MAGIC_STRINGS):
                    self.unsupported_columns.append(str_column)
                else:
                    self.supported_columns.append(str_column)

            with StepProfiler(method="rows.index", step="create the row group offsets"):
                self.row_group_offsets = np.cumsum(
                    [
                        parquet_file.metadata.row_group(group_id).num_rows
                        for parquet_file in parquet_files
                        for group_id in range(parquet_file.metadata.num_row_groups)
                    ]
                )
            with StepProfiler(method="rows.index", step="create the row group readers"):
                self.row_group_readers = [
                    partial(parquet_file.read_row_group, i=group_id, columns=self.supported_columns)
                    for parquet_file in parquet_files
                    for group_id in range(parquet_file.metadata.num_row_groups)
                ]

    # note that this cache size is global for the class, not per instance
    @lru_cache(maxsize=1024)
    def query(self, offset: int, length: int) -> pa.Table:
        """Query the parquet files

        Note that this implementation will always read at least one row group, to get the list of columns and always
        have the same schema, even if the requested rows are invalid (out of range).

        Args:
            offset (int): The first row to read.
            length (int): The number of rows to read.

        Returns:
            pa.Table: The requested rows.
        """
        if (len(self.row_group_offsets) == 0) or (len(self.row_group_readers) == 0):
            raise ParquetResponseEmptyError("No parquet files found.")
        last_row_in_parquet = self.row_group_offsets[-1] - 1
        first_row = min(offset, last_row_in_parquet)
        last_row = min(offset, offset + length - 1, last_row_in_parquet)
        first_row_group_id, last_row_group_id = np.searchsorted(
            self.row_group_offsets, [first_row, last_row], side="right"
        )
        pa_table = pa.concat_tables(
            [self.row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)]
        )
        first_row_in_pa_table = self.row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
        return pa_table.slice(offset - first_row_in_pa_table, length)


class Indexer:
    def __init__(
        self,
        config_parquet_processing_steps: List[ProcessingStep],
        init_processing_steps: List[ProcessingStep],
        hf_endpoint: str,
        hf_token: Optional[str] = None,
    ):
        self.config_parquet_processing_steps = config_parquet_processing_steps
        self.init_processing_steps = init_processing_steps
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token

    @lru_cache(maxsize=128)
    def get_rows_index(
        self,
        dataset: str,
        config: str,
        split: str,
    ) -> RowsIndex:
        return RowsIndex(
            dataset=dataset,
            config=config,
            split=split,
            config_parquet_processing_steps=self.config_parquet_processing_steps,
            init_processing_steps=self.init_processing_steps,
            hf_endpoint=self.hf_endpoint,
            hf_token=self.hf_token,
        )


Row = Mapping[str, Any]


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Row


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


class RowItem(TypedDict):
    row_idx: int
    row: Mapping[str, Any]
    truncated_cells: List[str]


def to_rows_list(
    pa_table: pa.Table,
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
    features: Features,
    unsupported_columns: List[str],
) -> List[RowItem]:
    num_rows = pa_table.num_rows
    for idx, (column, feature) in enumerate(features.items()):
        if column in unsupported_columns:
            pa_table = pa_table.add_column(idx, column, pa.array([None] * num_rows))
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(
            dataset=dataset,
            config=config,
            split=split,
            rows=pa_table.to_pylist(),
            features=features,
            cached_assets_base_url=cached_assets_base_url,
            cached_assets_directory=cached_assets_directory,
        )
    except Exception as err:
        raise ParquetDataProcessingError(
            "Server error while post-processing the split rows. Please report the issue."
        ) from err
    return [
        {
            "row_idx": idx + offset,
            "row": row,
            "truncated_cells": unsupported_columns,
        }
        for idx, row in enumerate(transformed_rows)
    ]


def transform_rows(
    dataset: str,
    config: str,
    split: str,
    rows: List[Row],
    features: Features,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
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
                assets_base_url=cached_assets_base_url,
                assets_directory=cached_assets_directory,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


def _greater_or_equal(row_dir_name: str, row_idx: int, on_error: bool) -> bool:
    try:
        return int(row_dir_name) >= row_idx
    except ValueError:
        return on_error


def clean_cached_assets(
    dataset: str,
    cached_assets_directory: StrPath,
    keep_rows_below_index: int,
    keep_n_most_recent_rows: int,
    max_clean_sample_size: int,
) -> None:
    row_directories = glob_rows_in_assets_dir(dataset, cached_assets_directory)
    row_directories_sample = list(
        islice(
            (
                row_dir
                for row_dir in row_directories
                if _greater_or_equal(row_dir.name, keep_rows_below_index, on_error=True)
            ),
            max_clean_sample_size + keep_n_most_recent_rows,
        )
    )
    if len(row_directories_sample) > keep_n_most_recent_rows:
        row_dirs_to_delete = sorted(row_directories_sample, key=os.path.getmtime, reverse=True)[
            keep_n_most_recent_rows:
        ]
        for row_dir_to_delete in row_dirs_to_delete:
            shutil.rmtree(row_dir_to_delete, ignore_errors=True)


def create_response(
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: List[str],
) -> Any:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(
            pa_table,
            dataset,
            config,
            split,
            cached_assets_base_url,
            cached_assets_directory,
            offset,
            features,
            unsupported_columns,
        ),
    }


def create_rows_endpoint(
    config_parquet_processing_steps: List[ProcessingStep],
    init_processing_steps: List[ProcessingStep],
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    indexer = Indexer(
        config_parquet_processing_steps=config_parquet_processing_steps,
        init_processing_steps=init_processing_steps,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
    )

    async def rows_endpoint(request: Request) -> Response:
        with StepProfiler(method="rows_endpoint", step="all"):
            try:
                with StepProfiler(method="rows_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    if not dataset or not config or not split or not are_valid_parameters([dataset, config, split]):
                        raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Offset must be positive")
                    length = int(request.query_params.get("length", MAX_ROWS))
                    if length < 0:
                        raise InvalidParameterError("Length must be positive")
                    if length > MAX_ROWS:
                        raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")
                    logging.info(
                        f"/rows, dataset={dataset}, config={config}, split={split}, offset={offset}, length={length}"
                    )
                with StepProfiler(method="rows_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                with StepProfiler(method="rows_endpoint", step="get row groups index"):
                    rows_index = indexer.get_rows_index(dataset=dataset, config=config, split=split)
                with StepProfiler(method="rows_endpoint", step="query the rows"):
                    pa_table = rows_index.query(offset=offset, length=length)
                with StepProfiler(method="rows_endpoint", step="clean cache"):
                    if random.random() < CLEAN_CACHE_PROBA:  # no need to do it every time
                        clean_cached_assets(
                            dataset=dataset,
                            cached_assets_directory=cached_assets_directory,
                            keep_rows_below_index=KEEP_ROWS_BELOW_INDEX,
                            keep_n_most_recent_rows=KEEP_N_MOST_RECENT_ROWS,
                            max_clean_sample_size=MAX_CLEAN_SAMPLE_SIZE,
                        )
                with StepProfiler(method="rows_endpoint", step="transform to a list"):
                    response = create_response(
                        dataset=dataset,
                        config=config,
                        split=split,
                        cached_assets_base_url=cached_assets_base_url,
                        cached_assets_directory=cached_assets_directory,
                        pa_table=pa_table,
                        offset=offset,
                        features=rows_index.features,
                        unsupported_columns=rows_index.unsupported_columns,
                    )
                with StepProfiler(method="rows_endpoint", step="update last modified time of rows in asset dir"):
                    update_last_modified_date_of_rows_in_assets_dir(
                        dataset=dataset,
                        config=config,
                        split=split,
                        offset=offset,
                        length=length,
                        assets_directory=cached_assets_directory,
                    )
                with StepProfiler(method="rows_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short)

    return rows_endpoint
