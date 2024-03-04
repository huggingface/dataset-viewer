import asyncio
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Optional, TypedDict, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Features, Value
from datasets.features.features import FeatureType
from datasets.utils.py_utils import size_str
from fsspec.implementations.http import HTTPFile, HTTPFileSystem
from huggingface_hub import HfFileSystem
from pyarrow.lib import ArrowInvalid

from libcommon.constants import CONFIG_PARQUET_METADATA_KIND
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.viewer_utils.features import get_supported_unsupported_columns

# For partial Parquet export we have paths like "en/partial-train/0000.parquet".
# "-" is not allowed is split names so we use it in the prefix to avoid collisions.
# We also use this prefix for the DuckDB index file name
PARTIAL_PREFIX = "partial-"

# For paths like "en/train-part0/0000.parquet", "en/train-part1/0000.parquet" up to "en/train-part9/9999.parquet".
# Note that "-" is forbidden for split names so it doesn't create directory names collisions.
PART_SUFFIX = "-part{}"


class EmptyParquetMetadataError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class FileSystemError(Exception):
    pass


class TooBigRows(Exception):
    pass


class SchemaMismatchError(Exception):
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
        parquet_file_url (`str`): URL of a Parquet file from the Parquet export
            It must be placed in a directory name after the split,
            e.g. "train" or "partial-train" for a partial split export.

            You can also pass the URL of any file in the directory since
            this function only checks the name of the parent directory.
            For example, it also works for DuckDB index files in the same
            directory as the Parquet files.

    Returns:
        `bool`: True is the Parquet export is partial,
            or False if it's the full dataset.
    """
    split_directory_name_for_parquet_export = extract_split_name_from_parquet_url(parquet_file_url)
    return split_directory_name_for_parquet_export.startswith(PARTIAL_PREFIX)


def extract_split_name_from_parquet_url(parquet_url: str) -> str:
    """
    Extracts the split name from a parquet file URL
    stored in the `refs/convert/parquet` branch of a
    dataset repository on the Hub

    Args:
        parquet_url (`str`): The URL to extract the split name from.

    Returns:
        `str`: The extracted split name.
    """
    split_name = parquet_url.rsplit("/", 2)[1]
    return split_name


def get_num_parquet_files_to_process(
    parquet_files: list[ParquetFileMetadataItem],
    parquet_metadata_directory: StrPath,
    max_size_bytes: int,
) -> tuple[int, int, int]:
    """
    For splits bigger than `max_size_bytes` computes the number of parquet files for future processing steps
    (for example, duckdb index, statistics), as well as number of bytes and number of rows in these files.
    Ensures that there is at least 1 parquet file to index.

    Returns:
        tuple of number of files, number of bytes and number of rows to process
    """
    num_parquet_files_to_process, num_bytes, num_rows = 0, 0, 0
    for parquet_file_id, parquet_file in enumerate(parquet_files):
        parquet_metadata_path = os.path.join(parquet_metadata_directory, parquet_file["parquet_metadata_subpath"])
        parquet_metadata = pq.read_metadata(parquet_metadata_path)
        num_parquet_files_to_process += 1
        num_rows += parquet_metadata.num_rows
        for row_group_id in range(parquet_metadata.num_row_groups):
            num_bytes += parquet_metadata.row_group(row_group_id).total_byte_size
        if num_bytes > max_size_bytes:
            break
    return num_parquet_files_to_process, num_bytes, num_rows


@dataclass
class RowGroupReader:
    parquet_file: pq.ParquetFile
    group_id: int

    def read(self, columns: list[str]) -> pa.Table:
        return self.parquet_file.read_row_group(i=self.group_id, columns=columns)

    def read_truncated_binary(self, columns: list[str], max_binary_length: int) -> tuple[pa.Table, list[str]]:
        pa_table = self.parquet_file.read_row_group(i=self.group_id, columns=columns)
        truncated_columns: list[str] = []
        if max_binary_length:
            for field_idx, field in enumerate(pa_table.schema):
                if field.type == pa.binary() and pa_table[field_idx].nbytes > max_binary_length:
                    truncated_array = pc.binary_slice(pa_table[field_idx], 0, max_binary_length // len(pa_table))
                    pa_table = pa_table.set_column(field_idx, field, truncated_array)
                    truncated_columns.append(field.name)
        return pa_table, truncated_columns

    def read_size(self, columns: Optional[Iterable[str]] = None) -> int:
        if columns is None:
            return self.parquet_file.metadata.row_group(self.group_id).total_byte_size  # type: ignore
        else:
            columns = set(columns)
            columns_metadata = self.parquet_file.metadata.row_group(self.group_id).to_dict()["columns"]
            return sum(
                column_metadata["total_uncompressed_size"]
                for column_metadata in columns_metadata
                if column_metadata["path_in_schema"] in columns
            )


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

    def query_truncated_binary(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        """Query the parquet files

        Note that this implementation will always read at least one row group, to get the list of columns and always
        have the same schema, even if the requested rows are invalid (out of range).

        This is the same as query() except that:

        - it computes a maximum size to allocate to binary data in step "parquet_index_with_metadata.row_groups_size_check_truncated_binary"
        - it uses `read_truncated_binary()` in step "parquet_index_with_metadata.query_truncated_binary".

        Args:
            offset (`int`): The first row to read.
            length (`int`): The number of rows to read.

        Raises:
            [`TooBigRows`]: if the arrow data from the parquet row groups is bigger than max_arrow_data_in_memory

        Returns:
            `pa.Table`: The requested rows.
            `list[strl]: List of truncated columns.
        """
        all_columns = set(self.features)
        binary_columns = set(column for column, feature in self.features.items() if feature == Value("binary"))
        if not binary_columns:
            return self.query(offset=offset, length=length), []
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
                return parquet_files[0].read(), []

            last_row_in_parquet = row_group_offsets[-1] - 1
            first_row = min(parquet_offset, last_row_in_parquet)
            last_row = min(parquet_offset + length - 1, last_row_in_parquet)

            first_row_group_id, last_row_group_id = np.searchsorted(
                row_group_offsets, [first_row, last_row], side="right"
            )

        with StepProfiler(
            method="parquet_index_with_metadata.row_groups_size_check_truncated_binary",
            step="check if the rows can fit in memory",
        ):
            in_memory_max_non_binary_size = sum(
                [
                    row_group_readers[i].read_size(columns=all_columns - binary_columns)
                    for i in range(first_row_group_id, last_row_group_id + 1)
                ]
            )
            in_memory_max_binary_size = max(
                [
                    row_group_readers[i].read_size(columns=binary_columns)
                    for i in range(first_row_group_id, last_row_group_id + 1)
                ]
            )
            in_memory_max_size = in_memory_max_non_binary_size + in_memory_max_binary_size
            if in_memory_max_size > self.max_arrow_data_in_memory:
                raise TooBigRows(
                    "Rows from parquet row groups are too big to be read:"
                    f" {size_str(in_memory_max_size)} (max={size_str(self.max_arrow_data_in_memory)})"
                )

        with StepProfiler(method="parquet_index_with_metadata.query_truncated_binary", step="read the row groups"):
            # This is a simple heuristic of how much we need to truncate binary data
            max_binary_length = max(
                int(
                    (self.max_arrow_data_in_memory - in_memory_max_non_binary_size)
                    / (last_row_group_id + 1 - first_row_group_id)
                    / len(binary_columns)
                    / 2  # we divide more in case the row groups are not evenly distributed
                ),
                20,
            )  # we use a minimum length to not end up with too empty cells
            try:
                pa_tables: list[pa.Table] = []
                truncated_columns: set[str] = set()
                for i in range(first_row_group_id, last_row_group_id + 1):
                    rg_pa_table, rg_truncated_columns = row_group_readers[i].read_truncated_binary(
                        self.supported_columns, max_binary_length=max_binary_length
                    )
                    pa_tables.append(rg_pa_table)
                    truncated_columns |= set(rg_truncated_columns)
                pa_table = pa.concat_tables(pa_tables)
            except ArrowInvalid as err:
                raise SchemaMismatchError("Parquet files have different schema.", err)
            first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
            return pa_table.slice(parquet_offset - first_row_in_pa_table, length), list(truncated_columns)

    def query(self, offset: int, length: int) -> pa.Table:
        """Query the parquet files

        Note that this implementation will always read at least one row group, to get the list of columns and always
        have the same schema, even if the requested rows are invalid (out of range).

        Args:
            offset (`int`): The first row to read.
            length (`int`): The number of rows to read.

        Raises:
            [`TooBigRows`]: if the arrow data from the parquet row groups is bigger than max_arrow_data_in_memory

        Returns:
            `pa.Table`: The requested rows.
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
            try:
                pa_table = pa.concat_tables(
                    [
                        row_group_readers[i].read(self.supported_columns)
                        for i in range(first_row_group_id, last_row_group_id + 1)
                    ]
                )
            except ArrowInvalid as err:
                raise SchemaMismatchError("Parquet files have different schema.", err)
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
            raise EmptyParquetMetadataError("No parquet files found.")

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
        httpfs: HfFileSystem,
        hf_token: Optional[str],
        parquet_metadata_directory: StrPath,
        max_arrow_data_in_memory: int,
        unsupported_features: list[FeatureType] = [],
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
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
                response = get_previous_step_or_raise(
                    kind=CONFIG_PARQUET_METADATA_KIND,
                    dataset=self.dataset,
                    config=self.config,
                    split=None,
                )
                self.revision = response["dataset_git_revision"]
                content = response["content"]
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
            offset (`int`): The first row to read.
            length (`int`): The number of rows to read.

        Returns:
            `pa.Table`: The requested rows.
        """
        logging.info(
            f"Query {type(self.parquet_index).__name__} for dataset={self.dataset}, config={self.config},"
            f" split={self.split}, offset={offset}, length={length}"
        )
        return self.parquet_index.query(offset=offset, length=length)

    # note that this cache size is global for the class, not per instance
    @lru_cache(maxsize=1)
    def query_truncated_binary(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        """Query the parquet files

        Note that this implementation will always read at least one row group, to get the list of columns and always
        have the same schema, even if the requested rows are invalid (out of range).

        Args:
            offset (`int`): The first row to read.
            length (`int`): The number of rows to read.

        Returns:
            `pa.Table`: The requested rows.
            `list[str]`: List of truncated columns.
        """
        logging.info(
            f"Query {type(self.parquet_index).__name__} for dataset={self.dataset}, config={self.config},"
            f" split={self.split}, offset={offset}, length={length}, with truncated binary"
        )
        return self.parquet_index.query_truncated_binary(offset=offset, length=length)


class Indexer:
    def __init__(
        self,
        parquet_metadata_directory: StrPath,
        httpfs: HTTPFileSystem,
        max_arrow_data_in_memory: int,
        unsupported_features: list[FeatureType] = [],
        all_columns_supported_datasets_allow_list: Union[Literal["all"], list[str]] = "all",
        hf_token: Optional[str] = None,
    ):
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
            httpfs=self.httpfs,
            hf_token=self.hf_token,
            parquet_metadata_directory=self.parquet_metadata_directory,
            max_arrow_data_in_memory=self.max_arrow_data_in_memory,
            unsupported_features=unsupported_features,
        )
