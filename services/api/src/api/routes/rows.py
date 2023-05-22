# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import asyncio
import logging
import os
import random
import shutil
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import islice
from os import PathLike
from typing import Any, Callable, List, Mapping, Optional, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features
from fsspec.implementations.http import HTTPFile, HTTPFileSystem
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import safe_quote
from libcommon.processing_graph import ProcessingGraph
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

logger = logging.getLogger(__name__)


httpfs = HTTPFileSystem()
session = asyncio.run(httpfs.set_session())


MAX_ROWS = 100

PARQUET_REVISION = "refs/convert/parquet"


StrPath = Union[str, PathLike[str]]


class FileSystemError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class ParquetResponseEmptyError(Exception):
    pass


class ParquetDataProcessingError(Exception):
    pass


class ParquetFileItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class ParquetFileMetadataItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int
    num_rows: int
    parquet_metadata_subpath: str


# TODO: how to invalidate the cache when the parquet branch is created or deleted?
@lru_cache(maxsize=128)
def get_hf_fs(hf_token: Optional[str]) -> HfFileSystem:
    """Get the Hugging Face filesystem.

    Args:
        hf_token (Optional[str]): The token to access the filesystem.
    Returns:
        HfFileSystem: The Hugging Face filesystem.
    """
    return HfFileSystem(token=hf_token)


def get_hf_parquet_uris(paths: List[str], dataset: str) -> List[str]:
    """Get the Hugging Face URIs from the Parquet branch of the dataset repository (see PARQUET_REVISION).

    Args:
        paths (List[str]): List of paths.
        dataset (str): The dataset name.
    Returns:
        List[str]: List of Parquet URIs.
    """
    return [f"hf://datasets/{dataset}@{safe_quote(PARQUET_REVISION)}/{path}" for path in paths]


UNSUPPORTED_FEATURES_MAGIC_STRINGS = ["'binary'"]


def get_supported_unsupported_columns(features: Features) -> Tuple[List[str], List[str]]:
    supported_columns, unsupported_columns = [], []
    for column, feature in features.items():
        str_feature = str(feature)
        str_column = str(column)
        if any(magic_string in str_feature for magic_string in UNSUPPORTED_FEATURES_MAGIC_STRINGS):
            unsupported_columns.append(str_column)
        else:
            supported_columns.append(str_column)
    return supported_columns, unsupported_columns


@dataclass
class ParquetIndexWithoutMetadata:
    features: Features
    supported_columns: List[str]
    unsupported_columns: List[str]
    row_group_offsets: npt.NDArray[np.int64]
    row_group_readers: List[Callable[[], pa.Table]]

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
        last_row = min(offset + length - 1, last_row_in_parquet)
        first_row_group_id, last_row_group_id = np.searchsorted(
            self.row_group_offsets, [first_row, last_row], side="right"
        )
        pa_table = pa.concat_tables(
            [self.row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)]
        )
        first_row_in_pa_table = self.row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
        return pa_table.slice(offset - first_row_in_pa_table, length)

    @staticmethod
    def from_parquet_file_items(
        parquet_file_items: List[ParquetFileItem], dataset: str, config: str, split: str, hf_token: Optional[str]
    ) -> "ParquetIndexWithoutMetadata":
        try:
            sources = sorted(f"{config}/{parquet_file['filename']}" for parquet_file in parquet_file_items)
        except Exception as e:
            raise ParquetResponseFormatError(f"Could not parse the list of parquet files: {e}") from e
        logging.debug(
            f"Found {len(sources)} parquet files for dataset={dataset}, config={config}, split={split}: {sources}"
        )
        if not sources:
            raise ParquetResponseEmptyError("No parquet files found.")
        with StepProfiler(method="rows.index.without_metadata", step="get the Hub's dataset filesystem"):
            fs = get_hf_fs(hf_token=hf_token)
        with StepProfiler(method="rows.index.without_metadata", step="get the source URIs"):
            source_uris = get_hf_parquet_uris(sources, dataset=dataset)
        with StepProfiler(method="rows.index.without_metadata", step="get one parquet reader per parquet file"):
            desc = f"{dataset}/{config}/{split}"
            try:
                parquet_files: List[pq.ParquetFile] = thread_map(
                    partial(pq.ParquetFile, filesystem=fs), source_uris, desc=desc, unit="pq", disable=True
                )
            except Exception as e:
                raise FileSystemError(f"Could not read the parquet files: {e}") from e
        with StepProfiler(method="rows.index.without_metadata", step="get the dataset's features"):
            features = Features.from_arrow_schema(parquet_files[0].schema.to_arrow_schema())
            supported_columns, unsupported_columns = get_supported_unsupported_columns(features)

        with StepProfiler(method="rows.index.without_metadata", step="create the row group offsets"):
            row_group_offsets = np.cumsum(
                [
                    parquet_file.metadata.row_group(group_id).num_rows
                    for parquet_file in parquet_files
                    for group_id in range(parquet_file.metadata.num_row_groups)
                ]
            )
        with StepProfiler(method="rows.index.without_metadata", step="create the row group readers"):
            row_group_readers: List[Callable[[], pa.Table]] = [
                partial(parquet_file.read_row_group, i=group_id, columns=supported_columns)
                for parquet_file in parquet_files
                for group_id in range(parquet_file.metadata.num_row_groups)
            ]
        return ParquetIndexWithoutMetadata(
            features=features,
            supported_columns=supported_columns,
            unsupported_columns=unsupported_columns,
            row_group_offsets=row_group_offsets,
            row_group_readers=row_group_readers,
        )


@dataclass
class ParquetIndexWithMetadata:
    features: Features
    supported_columns: List[str]
    unsupported_columns: List[str]
    parquet_files_urls: List[str]
    metadata_paths: List[str]
    num_bytes: List[int]
    num_rows: List[int]
    hf_token: Optional[str]

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
        with StepProfiler(
            method="rows.query.with_metadata", step="get the parquet files than contain the requested rows"
        ):
            parquet_file_offsets = np.cumsum(self.num_rows)

            last_row_in_parquet = parquet_file_offsets[-1] - 1
            first_row = min(offset, last_row_in_parquet)
            last_row = min(offset + length - 1, last_row_in_parquet)
            first_parquet_file_id, last_parquet_file_id = np.searchsorted(
                parquet_file_offsets, [first_row, last_row], side="right"
            )
            parquet_offset = (
                offset - parquet_file_offsets[first_parquet_file_id - 1] if first_parquet_file_id > 0 else offset
            )
            urls = self.parquet_files_urls[first_parquet_file_id : last_parquet_file_id + 1]  # noqa: E203
            metadata_paths = self.metadata_paths[first_parquet_file_id : last_parquet_file_id + 1]  # noqa: E203
            num_bytes = self.num_bytes[first_parquet_file_id : last_parquet_file_id + 1]  # noqa: E203

        with StepProfiler(
            method="rows.query.with_metadata", step="load the remote parquet files using metadata from disk"
        ):
            parquet_files = [
                pq.ParquetFile(
                    HTTPFile(httpfs, url, session=session, size=size, loop=httpfs.loop, cache_type=None),
                    metadata=pq.read_metadata(metadata_path),
                    pre_buffer=True,
                )
                for url, metadata_path, size in zip(urls, metadata_paths, num_bytes)
            ]

        with StepProfiler(
            method="rows.query.with_metadata", step="get the row groups than contain the requested rows"
        ):
            row_group_offsets = np.cumsum(
                [
                    parquet_file.metadata.row_group(group_id).num_rows
                    for parquet_file in parquet_files
                    for group_id in range(parquet_file.metadata.num_row_groups)
                ]
            )
            row_group_readers: List[Callable[[], pa.Table]] = [
                partial(parquet_file.read_row_group, i=group_id, columns=self.supported_columns)
                for parquet_file in parquet_files
                for group_id in range(parquet_file.metadata.num_row_groups)
            ]

            last_row_in_parquet = row_group_offsets[-1] - 1
            first_row = min(parquet_offset, last_row_in_parquet)
            last_row = min(parquet_offset + length - 1, last_row_in_parquet)

            first_row_group_id, last_row_group_id = np.searchsorted(
                row_group_offsets, [first_row, last_row], side="right"
            )

        with StepProfiler(method="rows.query.with_metadata", step="read the row groups"):
            pa_table = pa.concat_tables(
                [row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)]
            )
            first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
            return pa_table.slice(parquet_offset - first_row_in_pa_table, length)

    @staticmethod
    def from_parquet_metadata_items(
        parquet_file_metadata_items: List[ParquetFileMetadataItem],
        parquet_metadata_directory: StrPath,
        hf_token: Optional[str],
    ) -> "ParquetIndexWithMetadata":
        if not parquet_file_metadata_items:
            raise ParquetResponseEmptyError("No parquet files found.")

        with StepProfiler(method="rows.index", step="get the index from parquet metadata"):
            try:
                parquet_files_metadata = sorted(
                    parquet_file_metadata_items, key=lambda parquet_file_metadata: parquet_file_metadata["filename"]
                )
                parquet_files_urls = [parquet_file_metadata["url"] for parquet_file_metadata in parquet_files_metadata]
                metadata_paths = [
                    os.path.join(parquet_metadata_directory, parquet_file_metadata["parquet_metadata_subpath"])
                    for parquet_file_metadata in parquet_files_metadata
                ]
                num_bytes = [parquet_file_metadata["size"] for parquet_file_metadata in parquet_files_metadata]
                num_rows = [parquet_file_metadata["num_rows"] for parquet_file_metadata in parquet_files_metadata]
            except Exception as e:
                raise ParquetResponseFormatError(f"Could not parse the list of parquet files: {e}") from e

        with StepProfiler(method="rows.index.with_metadata", step="get the dataset's features"):
            features = Features.from_arrow_schema(pq.read_schema(metadata_paths[0]))
            supported_columns, unsupported_columns = get_supported_unsupported_columns(features)
        return ParquetIndexWithMetadata(
            features=features,
            supported_columns=supported_columns,
            unsupported_columns=unsupported_columns,
            parquet_files_urls=parquet_files_urls,
            metadata_paths=metadata_paths,
            num_bytes=num_bytes,
            num_rows=num_rows,
            hf_token=hf_token,
        )


class RowsIndex:
    def __init__(
        self,
        dataset: str,
        config: str,
        split: str,
        processing_graph: ProcessingGraph,
        hf_endpoint: str,
        hf_token: Optional[str],
        parquet_metadata_directory: StrPath,
    ):
        self.dataset = dataset
        self.revision: Optional[str] = None
        self.config = config
        self.split = split
        self.processing_graph = processing_graph
        self.parquet_index = self._init_parquet_index(
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    def _init_parquet_index(
        self,
        hf_endpoint: str,
        hf_token: Optional[str],
        parquet_metadata_directory: StrPath,
    ) -> Union[ParquetIndexWithMetadata, ParquetIndexWithoutMetadata]:
        with StepProfiler(method="rows.index", step="all"):
            # get the list of parquet files
            with StepProfiler(method="rows.index", step="get list of parquet files for split"):
                config_parquet_processing_steps = self.processing_graph.get_config_parquet_processing_steps()
                config_parquet_metadata_processing_steps = (
                    self.processing_graph.get_config_parquet_metadata_processing_steps()
                )
                if not config_parquet_processing_steps:
                    raise RuntimeError("No processing steps are configured to provide a config's parquet response.")
                try:
                    result = get_cache_entry_from_steps(
                        processing_steps=config_parquet_metadata_processing_steps + config_parquet_processing_steps,
                        dataset=self.dataset,
                        config=self.config,
                        split=None,
                        processing_graph=self.processing_graph,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                    )
                    self.revision = result["dataset_git_revision"]
                    content = result["content"]
                except ApiCustomError as e:
                    raise e
                except Exception as e:
                    raise UnexpectedError("Could not get the list of parquet files to fetch the rows from.") from e
                    # ^ TODO: improve the error, depending on the case
            if content and "parquet_files" in content:
                return ParquetIndexWithoutMetadata.from_parquet_file_items(
                    [
                        parquet_item
                        for parquet_item in content["parquet_files"]
                        if parquet_item["split"] == self.split and parquet_item["config"] == self.config
                    ],
                    dataset=self.dataset,
                    config=self.config,
                    split=self.split,
                    hf_token=hf_token,
                )
            else:
                return ParquetIndexWithMetadata.from_parquet_metadata_items(
                    [
                        parquet_item
                        for parquet_item in content["parquet_files_metadata"]
                        if parquet_item["split"] == self.split and parquet_item["config"] == self.config
                    ],
                    parquet_metadata_directory=parquet_metadata_directory,
                    hf_token=hf_token,
                )

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
        return self.parquet_index.query(offset=offset, length=length)


class Indexer:
    def __init__(
        self,
        processing_graph: ProcessingGraph,
        parquet_metadata_directory: StrPath,
        hf_endpoint: str,
        hf_token: Optional[str] = None,
    ):
        self.processing_graph = processing_graph
        self.parquet_metadata_directory = parquet_metadata_directory
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
            processing_graph=self.processing_graph,
            hf_endpoint=self.hf_endpoint,
            hf_token=self.hf_token,
            parquet_metadata_directory=self.parquet_metadata_directory,
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
            offset=offset,
        )
    except Exception as err:
        raise ParquetDataProcessingError(
            "Server error while post-processing the split rows. Please report the issue."
        ) from err
    return [
        {
            "row_idx": idx + offset,
            "row": row,
            "truncated_cells": [],
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
    offset: int,
) -> List[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=offset + row_idx,
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
    keep_first_rows_number: int,
    keep_most_recent_rows_number: int,
    max_cleaned_rows_number: int,
) -> None:
    """
    The cached assets directory is cleaned to save disk space using this simple (?) heuristic:

    1. it takes a big sample of rows from the cache using glob (max `max_cleaned_rows_number`)
    2. it keeps the most recent ones (max `keep_most_recent_rows_number`)
    3. it keeps the rows below a certain index (max `keep_first_rows_number`)
    4. it discards the rest

    To check for the most recent rows, it looks at the "last modified time" of rows directories.
    This time is updated every time a row is accessed using `update_last_modified_date_of_rows_in_assets_dir()`.

    Args:
        dataset (`str`):
            Dataset name e.g `squad` or `lhoestq/demo1`.
            Rows are cleaned in any dataset configuration or split of this dataset.
        cached_assets_directory (`str`):
            Directory containing the cached image and audio files
        keep_first_rows_number (`int`):
            Keep the rows with an index below a certain number
        keep_most_recent_rows_number (`int`):
            Keep the most recently accessed rows.
        max_cleaned_rows_number (`int`):
            Maximum number of rows to discard.
    """
    if keep_first_rows_number < 0 or keep_most_recent_rows_number < 0 or max_cleaned_rows_number < 0:
        raise ValueError(
            "Failed to run cached assets cleaning. Make sure all of keep_first_rows_number,"
            f" keep_most_recent_rows_number and max_cleaned_rows_number  are set (got {keep_first_rows_number},"
            f" {keep_most_recent_rows_number} and {max_cleaned_rows_number})"
        )
    row_directories = glob_rows_in_assets_dir(dataset, cached_assets_directory)
    row_directories_sample = list(
        islice(
            (
                row_dir
                for row_dir in row_directories
                if _greater_or_equal(row_dir.name, keep_first_rows_number, on_error=True)
            ),
            max_cleaned_rows_number + keep_most_recent_rows_number,
        )
    )
    if len(row_directories_sample) > keep_most_recent_rows_number:
        row_dirs_to_delete = sorted(row_directories_sample, key=os.path.getmtime, reverse=True)[
            keep_most_recent_rows_number:
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
    processing_graph: ProcessingGraph,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    parquet_metadata_directory: StrPath,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    clean_cache_proba: float = 0.0,
    keep_first_rows_number: int = -1,
    keep_most_recent_rows_number: int = -1,
    max_cleaned_rows_number: int = -1,
) -> Endpoint:
    indexer = Indexer(
        processing_graph=processing_graph,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        parquet_metadata_directory=parquet_metadata_directory,
    )

    async def rows_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
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
                    revision = rows_index.revision
                with StepProfiler(method="rows_endpoint", step="query the rows"):
                    pa_table = rows_index.query(offset=offset, length=length)
                with StepProfiler(method="rows_endpoint", step="clean cache"):
                    # no need to do it every time
                    if random.random() < clean_cache_proba:  # nosec
                        if (
                            keep_first_rows_number < 0
                            and keep_most_recent_rows_number < 0
                            and max_cleaned_rows_number < 0
                        ):
                            logger.debug(
                                "Params keep_first_rows_number, keep_most_recent_rows_number and"
                                " max_cleaned_rows_number are not set. Skipping cached assets cleaning."
                            )
                        else:
                            clean_cached_assets(
                                dataset=dataset,
                                cached_assets_directory=cached_assets_directory,
                                keep_first_rows_number=keep_first_rows_number,
                                keep_most_recent_rows_number=keep_most_recent_rows_number,
                                max_cleaned_rows_number=max_cleaned_rows_number,
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
                        features=rows_index.parquet_index.features,
                        unsupported_columns=rows_index.parquet_index.unsupported_columns,
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
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return rows_endpoint
