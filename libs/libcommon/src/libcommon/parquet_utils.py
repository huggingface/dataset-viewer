import asyncio
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Optional, TypedDict, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features
from datasets.features.features import FeatureType
from datasets.utils.py_utils import size_str
from fsspec.implementations.http import HTTPFile, HTTPFileSystem
from huggingface_hub import HfFileSystem

from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.viewer_utils.features import get_supported_unsupported_columns

# For partial Parquet export we have paths like "en/partial-train/0000.parquet".
# "-" is not allowed is split names so we use it in the prefix to avoid collisions.
# We also use this prefix for the DuckDB index file name
PARTIAL_PREFIX = "partial-"


class ParquetResponseEmptyError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class FileSystemError(Exception):
    pass


class TooBigRows(Exception):
    pass


class ParquetFileMetadataItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int
    num_rows: int
    parquet_metadata_subpath: str


def parquet_export_is_partial(parquet_file_url: str) -> bool:
    """
    Check if the Parquet export is the full dataset or if it's partial.
    The check is based on the split directory name from the URL.
    If it starts with "partial-" then it is a partially exported split.

    Args:
        parquet_file_url (str): URL of a Parquet file from the Parquet export
            It must be placed in a directory name after the split,
            e.g. "train" or "partial-train" for a partial split export.

            You can also pass the URL of any file in the directory since
            this function only checks the name of the parent directory.
            For example it also works for DuckDB index files in the same
            directory as the Parquet files.

    Returns:
        partial (bool): True is the Parquet export is partial,
            or False if it's the full dataset.
    """
    split_directory_name_for_parquet_export = parquet_file_url.rsplit("/", 2)[1]
    return split_directory_name_for_parquet_export.startswith(PARTIAL_PREFIX)


@dataclass
class RowGroupReader:
    parquet_file: pq.ParquetFile
    group_id: int

    def read(self, columns: list[str]) -> pa.Table:
        return self.parquet_file.read_row_group(i=self.group_id, columns=columns)

    def read_size(self) -> int:
        return self.parquet_file.metadata.row_group(self.group_id).total_byte_size  # type: ignore


@dataclass
class ParquetIndexWithMetadata:
    features: Features
    supported_columns: list[str]
    unsupported_columns: list[str]
    parquet_files_urls: list[str]
    metadata_paths: list[str]
    num_bytes: list[int]
    num_rows: list[int]
    httpfs: HTTPFileSystem
    hf_token: Optional[str]
    max_arrow_data_in_memory: int
    partial: bool

    num_rows_total: int = field(init=False)

    def __post_init__(self) -> None:
        if self.httpfs._session is None:
            self.httpfs_session = asyncio.run(self.httpfs.set_session())
        else:
            self.httpfs_session = self.httpfs._session
        self.num_rows_total = sum(self.num_rows)

    def query(self, offset: int, length: int) -> pa.Table:
        """Query the parquet files

        Note that this implementation will always read at least one row group, to get the list of columns and always
        have the same schema, even if the requested rows are invalid (out of range).

        Args:
            offset (int): The first row to read.
            length (int): The number of rows to read.

        Returns:
            pa.Table: The requested rows.

        Raises:
            TooBigRows: if the arrow data from the parquet row groups is bigger than max_arrow_data_in_memory
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
                        **self.httpfs.kwargs,
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
            row_group_readers = [
                RowGroupReader(parquet_file=parquet_file, group_id=group_id)
                for parquet_file in parquet_files
                for group_id in range(parquet_file.metadata.num_row_groups)
            ]

            if len(row_group_offsets) == 0 or row_group_offsets[-1] == 0:  # if the dataset is empty
                if offset < 0:
                    raise IndexError("Offset must be non-negative")
                return parquet_files[0].read()

            last_row_in_parquet = row_group_offsets[-1] - 1
            first_row = min(parquet_offset, last_row_in_parquet)
            last_row = min(parquet_offset + length - 1, last_row_in_parquet)

            first_row_group_id, last_row_group_id = np.searchsorted(
                row_group_offsets, [first_row, last_row], side="right"
            )

        with StepProfiler(
            method="parquet_index_with_metadata.row_groups_size_check", step="check if the rows can fit in memory"
        ):
            row_groups_size = sum(
                [row_group_readers[i].read_size() for i in range(first_row_group_id, last_row_group_id + 1)]
            )
            if row_groups_size > self.max_arrow_data_in_memory:
                raise TooBigRows(
                    "Rows from parquet row groups are too big to be read:"
                    f" {size_str(row_groups_size)} (max={size_str(self.max_arrow_data_in_memory)})"
                )

        with StepProfiler(method="parquet_index_with_metadata.query", step="read the row groups"):
            pa_table = pa.concat_tables(
                [
                    row_group_readers[i].read(self.supported_columns)
                    for i in range(first_row_group_id, last_row_group_id + 1)
                ]
            )
            first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
            return pa_table.slice(parquet_offset - first_row_in_pa_table, length)

    @staticmethod
    def from_parquet_metadata_items(
        parquet_file_metadata_items: list[ParquetFileMetadataItem],
        features: Optional[Features],
        parquet_metadata_directory: StrPath,
        httpfs: HTTPFileSystem,
        hf_token: Optional[str],
        max_arrow_data_in_memory: int,
        unsupported_features: list[FeatureType] = [],
    ) -> "ParquetIndexWithMetadata":
        if not parquet_file_metadata_items:
            raise ParquetResponseEmptyError("No parquet files found.")

        partial = parquet_export_is_partial(parquet_file_metadata_items[0]["url"])

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
            if features is None:  # config-parquet version<6 didn't have features
                features = Features.from_arrow_schema(pq.read_schema(metadata_paths[0]))
            supported_columns, unsupported_columns = get_supported_unsupported_columns(
                features,
                unsupported_features=unsupported_features,
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
            max_arrow_data_in_memory=max_arrow_data_in_memory,
            partial=partial,
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
        max_arrow_data_in_memory: int,
        unsupported_features: list[FeatureType] = [],
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.processing_graph = processing_graph
        self.httpfs = httpfs
        self.parquet_index = self._init_parquet_index(
            hf_token=hf_token,
            parquet_metadata_directory=parquet_metadata_directory,
            max_arrow_data_in_memory=max_arrow_data_in_memory,
            unsupported_features=unsupported_features,
        )

    def _init_parquet_index(
        self,
        hf_token: Optional[str],
        parquet_metadata_directory: StrPath,
        max_arrow_data_in_memory: int,
        unsupported_features: list[FeatureType] = [],
    ) -> ParquetIndexWithMetadata:
        with StepProfiler(method="rows_index._init_parquet_index", step="all"):
            # get the list of parquet files
            with StepProfiler(method="rows_index._init_parquet_index", step="get list of parquet files for split"):
                config_parquet_metadata_processing_steps = (
                    self.processing_graph.get_config_parquet_metadata_processing_steps()
                )
                cache_kinds = [step.cache_kind for step in config_parquet_metadata_processing_steps]
                result = get_previous_step_or_raise(
                    kinds=cache_kinds,
                    dataset=self.dataset,
                    config=self.config,
                    split=None,
                )
                self.revision = result.response["dataset_git_revision"]
                content = result.response["content"]
                if content.get("features"):  # config-parquet-metadata version<2 didn't have features
                    features = Features.from_dict(content["features"])
                else:
                    features = None
            logging.info(
                f"Create ParquetIndexWithMetadata for dataset={self.dataset}, config={self.config}, split={self.split}"
            )
            return ParquetIndexWithMetadata.from_parquet_metadata_items(
                [
                    parquet_item
                    for parquet_item in content["parquet_files_metadata"]
                    if parquet_item["split"] == self.split and parquet_item["config"] == self.config
                ],
                features=features,
                parquet_metadata_directory=parquet_metadata_directory,
                httpfs=self.httpfs,
                hf_token=hf_token,
                max_arrow_data_in_memory=max_arrow_data_in_memory,
                unsupported_features=unsupported_features,
            )

    # note that this cache size is global for the class, not per instance
    @lru_cache(maxsize=1)
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
        logging.info(
            f"Query {type(self.parquet_index).__name__} for dataset={self.dataset}, config={self.config},"
            f" split={self.split}, offset={offset}, length={length}"
        )
        return self.parquet_index.query(offset=offset, length=length)


class Indexer:
    def __init__(
        self,
        processing_graph: ProcessingGraph,
        parquet_metadata_directory: StrPath,
        httpfs: HTTPFileSystem,
        max_arrow_data_in_memory: int,
        unsupported_features: list[FeatureType] = [],
        all_columns_supported_datasets_allow_list: Union[Literal["all"], list[str]] = "all",
        hf_token: Optional[str] = None,
    ):
        self.processing_graph = processing_graph
        self.parquet_metadata_directory = parquet_metadata_directory
        self.httpfs = httpfs
        self.hf_token = hf_token
        self.max_arrow_data_in_memory = max_arrow_data_in_memory
        self.unsupported_features = unsupported_features
        self.all_columns_supported_datasets_allow_list = all_columns_supported_datasets_allow_list

    @lru_cache(maxsize=1)
    def get_rows_index(
        self,
        dataset: str,
        config: str,
        split: str,
    ) -> RowsIndex:
        filter_features = (
            self.all_columns_supported_datasets_allow_list != "all"
            and dataset not in self.all_columns_supported_datasets_allow_list
        )
        unsupported_features = self.unsupported_features if filter_features else []
        return RowsIndex(
            dataset=dataset,
            config=config,
            split=split,
            processing_graph=self.processing_graph,
            httpfs=self.httpfs,
            hf_token=self.hf_token,
            parquet_metadata_directory=self.parquet_metadata_directory,
            max_arrow_data_in_memory=self.max_arrow_data_in_memory,
            unsupported_features=unsupported_features,
        )
