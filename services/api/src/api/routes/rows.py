# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from functools import lru_cache, partial
from typing import Any, List, Mapping, Optional, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features
from hffs.fs import HfFileSystem
from libcommon.processing_graph import ProcessingStep
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


class FileSystemError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class ParquetResponseEmptyError(Exception):
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


UNSUPPORTED_FEATURES_MAGIC_STRINGS = ["Image(", "Audio(", "'binary'"]


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


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Mapping[str, Any]


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


def to_rows_list(pa_table: pa.Table, offset: int, features: Features, unsupported_columns: List[str]) -> List[RowItem]:
    num_rows = pa_table.num_rows
    for idx, (column, feature) in enumerate(features.items()):
        if column in unsupported_columns:
            pa_table = pa_table.add_column(idx, column, pa.array([None] * num_rows))
    return [
        {
            "row_idx": idx + offset,
            "row": row,
            "truncated_cells": unsupported_columns,
        }
        for idx, row in enumerate(pa_table.to_pylist())
    ]


def create_response(pa_table: pa.Table, offset: int, features: Features, unsupported_columns: List[str]) -> Any:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(pa_table, offset, features, unsupported_columns),
    }


def create_rows_endpoint(
    config_parquet_processing_steps: List[ProcessingStep],
    init_processing_steps: List[ProcessingStep],
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
                        dataset,
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
                with StepProfiler(method="rows_endpoint", step="transform to a list"):
                    response = create_response(pa_table, offset, rows_index.features, rows_index.unsupported_columns)
                with StepProfiler(method="rows_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short)

    return rows_endpoint
