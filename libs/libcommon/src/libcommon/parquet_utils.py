import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from os import PathLike
from typing import Callable, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features
from fsspec.implementations.http import HTTPFile, HTTPFileSystem
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import safe_quote
from tqdm.contrib.concurrent import thread_map

from libcommon.constants import PARQUET_REVISION
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise

StrPath = Union[str, PathLike[str]]


class ParquetResponseEmptyError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class FileSystemError(Exception):
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


def get_supported_unsupported_columns(
    features: Features,
    unsupported_features_magic_strings: List[str] = [],
) -> Tuple[List[str], List[str]]:
    supported_columns, unsupported_columns = [], []

    for column, feature in features.items():
        str_feature = str(feature)
        str_column = str(column)
        if any(magic_string in str_feature for magic_string in unsupported_features_magic_strings):
            unsupported_columns.append(str_column)
        else:
            supported_columns.append(str_column)
    return supported_columns, unsupported_columns


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
        parquet_file_items: List[ParquetFileItem],
        dataset: str,
        config: str,
        split: str,
        hf_token: Optional[str],
        unsupported_features_magic_strings: List[str] = [],
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
        with StepProfiler(
            method="parquet_index_without_metadata.from_parquet_file_items", step="get the Hub's dataset filesystem"
        ):
            fs = get_hf_fs(hf_token=hf_token)
        with StepProfiler(method="parquet_index_without_metadata.from_parquet_file_items", step="get the source URIs"):
            source_uris = get_hf_parquet_uris(sources, dataset=dataset)
        with StepProfiler(
            method="parquet_index_without_metadata.from_parquet_file_items",
            step="get one parquet reader per parquet file",
        ):
            desc = f"{dataset}/{config}/{split}"
            try:
                parquet_files: List[pq.ParquetFile] = thread_map(
                    partial(pq.ParquetFile, filesystem=fs), source_uris, desc=desc, unit="pq", disable=True
                )
            except Exception as e:
                raise FileSystemError(f"Could not read the parquet files: {e}") from e
        with StepProfiler(
            method="parquet_index_without_metadata.from_parquet_file_items", step="get the dataset's features"
        ):
            features = Features.from_arrow_schema(parquet_files[0].schema.to_arrow_schema())
            supported_columns, unsupported_columns = get_supported_unsupported_columns(
                features, unsupported_features_magic_strings=unsupported_features_magic_strings
            )

        with StepProfiler(
            method="parquet_index_without_metadata.from_parquet_file_items", step="create the row group offsets"
        ):
            row_group_offsets = np.cumsum(
                [
                    parquet_file.metadata.row_group(group_id).num_rows
                    for parquet_file in parquet_files
                    for group_id in range(parquet_file.metadata.num_row_groups)
                ]
            )
        with StepProfiler(
            method="parquet_index_without_metadata.from_parquet_file_items", step="create the row group readers"
        ):
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
    httpfs: HTTPFileSystem
    hf_token: Optional[str]

    def __post_init__(self) -> None:
        if self.httpfs._session is None:
            self.httpfs_session = asyncio.run(self.httpfs.set_session())
        else:
            self.httpfs_session = self.httpfs._session

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
            method="parquet_index_with_metadata.query", step="get the parquet files than contain the requested rows"
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
            method="parquet_index_with_metadata.query", step="load the remote parquet files using metadata from disk"
        ):
            parquet_files = [
                pq.ParquetFile(
                    HTTPFile(
                        self.httpfs,
                        url,
                        session=self.httpfs_session,
                        size=size,
                        loop=self.httpfs.loop,
                        cache_type=None,
                    ),
                    metadata=pq.read_metadata(metadata_path),
                    pre_buffer=True,
                )
                for url, metadata_path, size in zip(urls, metadata_paths, num_bytes)
            ]

        with StepProfiler(
            method="parquet_index_with_metadata.query", step="get the row groups than contain the requested rows"
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

        with StepProfiler(method="parquet_index_with_metadata.query", step="read the row groups"):
            pa_table = pa.concat_tables(
                [row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)]
            )
            first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
            return pa_table.slice(parquet_offset - first_row_in_pa_table, length)

    @staticmethod
    def from_parquet_metadata_items(
        parquet_file_metadata_items: List[ParquetFileMetadataItem],
        parquet_metadata_directory: StrPath,
        httpfs: HTTPFileSystem,
        hf_token: Optional[str],
        unsupported_features_magic_strings: List[str] = [],
    ) -> "ParquetIndexWithMetadata":
        if not parquet_file_metadata_items:
            raise ParquetResponseEmptyError("No parquet files found.")

        with StepProfiler(
            method="parquet_index_with_metadata.from_parquet_metadata_items",
            step="get the index from parquet metadata",
        ):
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

        with StepProfiler(
            method="parquet_index_with_metadata.from_parquet_metadata_items", step="get the dataset's features"
        ):
            features = Features.from_arrow_schema(pq.read_schema(metadata_paths[0]))
            supported_columns, unsupported_columns = get_supported_unsupported_columns(
                features,
                unsupported_features_magic_strings=unsupported_features_magic_strings,
            )
        return ParquetIndexWithMetadata(
            features=features,
            supported_columns=supported_columns,
            unsupported_columns=unsupported_columns,
            parquet_files_urls=parquet_files_urls,
            metadata_paths=metadata_paths,
            num_bytes=num_bytes,
            num_rows=num_rows,
            httpfs=httpfs,
            hf_token=hf_token,
        )


class RowsIndex:
    def __init__(
        self,
        dataset: str,
        config: str,
        split: str,
        processing_graph: ProcessingGraph,
        httpfs: HfFileSystem,
        hf_token: Optional[str],
        parquet_metadata_directory: StrPath,
        unsupported_features_magic_strings: List[str] = [],
    ):
        self.dataset = dataset
        self.revision: Optional[str] = None
        self.config = config
        self.split = split
        self.processing_graph = processing_graph
        self.httpfs = httpfs
        self.parquet_index = self._init_parquet_index(
            hf_token=hf_token,
            parquet_metadata_directory=parquet_metadata_directory,
            unsupported_features_magic_strings=unsupported_features_magic_strings,
        )

    def _init_parquet_index(
        self,
        hf_token: Optional[str],
        parquet_metadata_directory: StrPath,
        unsupported_features_magic_strings: List[str] = [],
    ) -> Union[ParquetIndexWithMetadata, ParquetIndexWithoutMetadata]:
        with StepProfiler(method="rows_index._init_parquet_index", step="all"):
            # get the list of parquet files
            with StepProfiler(method="rows_index._init_parquet_index", step="get list of parquet files for split"):
                config_parquet_processing_steps = self.processing_graph.get_config_parquet_processing_steps()
                if not config_parquet_processing_steps:
                    raise RuntimeError("No processing steps are configured to provide a config's parquet response.")

                config_parquet_metadata_processing_steps = (
                    self.processing_graph.get_config_parquet_metadata_processing_steps()
                )

                cache_kinds = [step.cache_kind for step in config_parquet_metadata_processing_steps]
                cache_kinds.extend([step.cache_kind for step in config_parquet_processing_steps])

                result = get_previous_step_or_raise(
                    kinds=cache_kinds,
                    dataset=self.dataset,
                    config=self.config,
                    split=None,
                )
                self.revision = result.response["dataset_git_revision"]
                content = result.response["content"]
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
                    unsupported_features_magic_strings=unsupported_features_magic_strings,
                )
            else:
                return ParquetIndexWithMetadata.from_parquet_metadata_items(
                    [
                        parquet_item
                        for parquet_item in content["parquet_files_metadata"]
                        if parquet_item["split"] == self.split and parquet_item["config"] == self.config
                    ],
                    parquet_metadata_directory=parquet_metadata_directory,
                    httpfs=self.httpfs,
                    hf_token=hf_token,
                    unsupported_features_magic_strings=unsupported_features_magic_strings,
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
        httpfs: HTTPFileSystem,
        unsupported_features_magic_strings: List[str] = [],
        all_columns_supported_datasets_allow_list: Union[Literal["all"], List[str]] = "all",
        hf_token: Optional[str] = None,
    ):
        self.processing_graph = processing_graph
        self.parquet_metadata_directory = parquet_metadata_directory
        self.httpfs = httpfs
        self.hf_token = hf_token
        self.unsupported_features_magic_strings = unsupported_features_magic_strings
        self.all_columns_supported_datasets_allow_list = all_columns_supported_datasets_allow_list

    @lru_cache(maxsize=128)
    def get_rows_index(
        self,
        dataset: str,
        config: str,
        split: str,
    ) -> RowsIndex:
        filter_magic_strings = (
            self.all_columns_supported_datasets_allow_list != "all"
            and dataset not in self.all_columns_supported_datasets_allow_list
        )
        unsupported_features_magic_strings = self.unsupported_features_magic_strings if filter_magic_strings else []
        return RowsIndex(
            dataset=dataset,
            config=config,
            split=split,
            processing_graph=self.processing_graph,
            httpfs=self.httpfs,
            hf_token=self.hf_token,
            parquet_metadata_directory=self.parquet_metadata_directory,
            unsupported_features_magic_strings=unsupported_features_magic_strings,
        )
