import asyncio
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Features, Value
from datasets.table import cast_table_to_schema
from datasets.utils.py_utils import size_str
from fsspec.implementations.http import HTTPFile, HTTPFileSystem
from pyarrow.lib import ArrowInvalid

from libcommon.constants import CONFIG_PARQUET_METADATA_KIND
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath

try:
    import libviewer as lv  # type: ignore

    _has_libviewer = True
except ImportError:
    _has_libviewer = False

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
    split_directory_name_for_parquet_export = extract_split_directory_from_parquet_url(parquet_file_url)
    return split_directory_name_for_parquet_export.startswith(PARTIAL_PREFIX)


def extract_split_directory_from_parquet_url(parquet_url: str) -> str:
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


def is_list_pa_type(parquet_file_path: Path, feature_name: str) -> bool:
    # Check if (Sequence) feature is internally a List, because it can also be Struct in datasets<4, see
    # https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Features
    feature_arrow_type = pq.read_schema(parquet_file_path).field(feature_name).type
    is_list: bool = pa.types.is_list(feature_arrow_type) or pa.types.is_large_list(feature_arrow_type)
    return is_list


def truncate_binary_columns(table: pa.Table, max_binary_length: int, features: Features) -> tuple[pa.Table, list[str]]:
    # truncate binary columns in the Arrow table to the specified maximum length
    # return a new Arrow table and the list of truncated columns
    if max_binary_length < 0:
        return table, []

    columns: dict[str, pa.Array] = {}
    truncated_column_names: list[str] = []
    for field_idx, field in enumerate(table.schema):  # noqa: F402
        if features[field.name] == Value("binary") and table[field_idx].nbytes > max_binary_length:
            truncated_array = pc.binary_slice(table[field_idx], 0, max_binary_length // len(table))
            columns[field.name] = truncated_array
            truncated_column_names.append(field.name)
        else:
            columns[field.name] = table[field_idx]

    return pa.table(columns), truncated_column_names


@dataclass
class RowGroupReader:
    parquet_file: pq.ParquetFile
    group_id: int
    schema: pa.Schema

    def read(self, columns: list[str]) -> pa.Table:
        if not set(self.parquet_file.schema_arrow.names) <= set(columns):
            raise SchemaMismatchError(
                f"Parquet files have different columns: {sorted(columns)} and {sorted(self.parquet_file.schema_arrow.names)}"
            )
        pa_table = self.parquet_file.read_row_group(i=self.group_id, columns=columns)
        # cast_table_to_schema adds null values to missing columns
        return cast_table_to_schema(pa_table, self.schema)

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
    files: list[ParquetFileMetadataItem]
    features: Features
    httpfs: HTTPFileSystem
    max_arrow_data_in_memory: int
    metadata_dir: Path

    file_offsets: np.ndarray = field(init=False)
    num_rows_total: int = field(init=False)
    partial: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.httpfs._session is None:
            self.httpfs_session = asyncio.run(self.httpfs.set_session())
        else:
            self.httpfs_session = self.httpfs._session

        num_rows = np.array([f["num_rows"] for f in self.files])
        self.file_offsets = np.cumsum(num_rows)
        self.num_rows_total = np.sum(num_rows)
        self.partial = parquet_export_is_partial(self.files[0]["url"])

    def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        """Query the parquet files

        Note that this implementation will always read at least one row group, to get the list of columns and always
        have the same schema, even if the requested rows are invalid (out of range).

        If binary columns are present, then:
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
        with StepProfiler(
            method="parquet_index_with_metadata.query", step="get the parquet files than contain the requested rows"
        ):
            last_row_in_parquet = self.file_offsets[-1] - 1
            first_row = min(offset, last_row_in_parquet)
            last_row = min(offset + length - 1, last_row_in_parquet)
            first_parquet_file_id, last_parquet_file_id = np.searchsorted(
                self.file_offsets, [first_row, last_row], side="right"
            )
            parquet_offset = (
                offset - self.file_offsets[first_parquet_file_id - 1] if first_parquet_file_id > 0 else offset
            )
            files_to_scan = self.files[first_parquet_file_id : last_parquet_file_id + 1]  # noqa: E203

        with StepProfiler(
            method="parquet_index_with_metadata.query", step="load the remote parquet files using metadata from disk"
        ):
            parquet_files = [
                pq.ParquetFile(
                    HTTPFile(
                        self.httpfs,
                        f["url"],
                        session=self.httpfs_session,
                        size=f["size"],
                        loop=self.httpfs.loop,
                        cache_type=None,
                        **self.httpfs.kwargs,
                    ),
                    metadata=pq.read_metadata(self.metadata_dir / f["parquet_metadata_subpath"]),
                    pre_buffer=True,
                )
                for f in files_to_scan
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
                RowGroupReader(parquet_file=parquet_file, group_id=group_id, schema=self.features.arrow_schema)
                for parquet_file in parquet_files
                for group_id in range(parquet_file.metadata.num_row_groups)
            ]

            if len(row_group_offsets) == 0 or row_group_offsets[-1] == 0:  # if the dataset is empty
                if offset < 0:
                    raise IndexError("Offset must be non-negative")
                return cast_table_to_schema(parquet_files[0].read(), self.features.arrow_schema), []

            last_row_in_parquet = row_group_offsets[-1] - 1
            first_row = min(parquet_offset, last_row_in_parquet)
            last_row = min(parquet_offset + length - 1, last_row_in_parquet)

            first_row_group_id, last_row_group_id = np.searchsorted(
                row_group_offsets, [first_row, last_row], side="right"
            )

        all_columns = set(self.features)
        binary_columns = set(column for column, feature in self.features.items() if feature == Value("binary"))
        if binary_columns:
            pa_table, truncated_columns = self._read_with_binary(
                row_group_readers, first_row_group_id, last_row_group_id, all_columns, binary_columns
            )
        else:
            pa_table, truncated_columns = self._read_without_binary(
                row_group_readers, first_row_group_id, last_row_group_id
            )

        first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
        return pa_table.slice(parquet_offset - first_row_in_pa_table, length), truncated_columns

    def _read_with_binary(
        self,
        row_group_readers: list[RowGroupReader],
        first_row_group_id: int,
        last_row_group_id: int,
        all_columns: set[str],
        binary_columns: set[str],
    ) -> tuple[pa.Table, list[str]]:
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
                columns = list(self.features.keys())
                truncated_columns: set[str] = set()
                for i in range(first_row_group_id, last_row_group_id + 1):
                    rg_pa_table = row_group_readers[i].read(columns)
                    rg_pa_table, rg_truncated_columns = truncate_binary_columns(
                        rg_pa_table, max_binary_length, self.features
                    )
                    pa_tables.append(rg_pa_table)
                    truncated_columns |= set(rg_truncated_columns)
                pa_table = pa.concat_tables(pa_tables)
            except ArrowInvalid as err:
                raise SchemaMismatchError("Parquet files have different schema.", err)

        return pa_table, list(truncated_columns)

    def _read_without_binary(
        self, row_group_readers: list[RowGroupReader], first_row_group_id: int, last_row_group_id: int
    ) -> tuple[pa.Table, list[str]]:
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
            columns = list(self.features.keys())
            try:
                pa_table = pa.concat_tables(
                    [row_group_readers[i].read(columns) for i in range(first_row_group_id, last_row_group_id + 1)]
                )
            except ArrowInvalid as err:
                raise SchemaMismatchError("Parquet files have different schema.", err)

        return pa_table, []


class RowsIndex:
    def __init__(
        self,
        dataset: str,
        config: str,
        split: str,
        httpfs: HTTPFileSystem,
        parquet_metadata_directory: StrPath,
        max_arrow_data_in_memory: int,
        max_scan_size: int,
        hf_token: Optional[str] = None,
        data_store: Optional[str] = None,
        use_libviewer_for_datasets: bool | set[str] = True,
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.httpfs = httpfs
        self.max_scan_size = max_scan_size
        self.use_libviewer_for_datasets = use_libviewer_for_datasets
        self.parquet_metadata_directory = Path(parquet_metadata_directory)

        if not isinstance(self.use_libviewer_for_datasets, (bool, set)):
            raise ValueError("`use_libviewer_for_datasets` must be a boolean or a set of dataset names")

        self._init_dataset_info()
        self._init_parquet_index(httpfs=httpfs, max_arrow_data_in_memory=max_arrow_data_in_memory)
        if _has_libviewer:
            self._init_viewer_index(hf_token=hf_token, data_store=data_store)

    def _init_dataset_info(self) -> None:
        # get the list of parquet files and features
        with StepProfiler(method="rows_index._get_dataset_metadata", step="all"):
            response = get_previous_step_or_raise(
                kind=CONFIG_PARQUET_METADATA_KIND, dataset=self.dataset, config=self.config, split=None
            )

        # set the revision of the dataset
        self.revision = response["dataset_git_revision"]

        # get the list of parquet files
        parquet_files = response["content"]["parquet_files_metadata"]
        # filter only files for the current split and config
        parquet_files = [f for f in parquet_files if f["split"] == self.split and f["config"] == self.config]
        # sort by filename to have a deterministic order
        parquet_files = sorted(parquet_files, key=lambda x: x["filename"])
        if not parquet_files:
            raise EmptyParquetMetadataError("No parquet files found.")
        self.parquet_files = parquet_files

        # retrieve the features from the mongo response
        features = response["content"].get("features")
        if features:
            self.features = Features.from_dict(features)
        else:
            # config-parquet version<6 didn't have features
            first_metadata_file = self.parquet_metadata_directory / parquet_files[0]["parquet_metadata_subpath"]
            arrow_schema = pq.read_schema(first_metadata_file)
            self.features = Features.from_arrow_schema(arrow_schema)

    def _init_parquet_index(
        self,
        httpfs: HTTPFileSystem,
        max_arrow_data_in_memory: int,
    ) -> None:
        logging.info(
            f"Create ParquetIndexWithMetadata for dataset={self.dataset}, config={self.config}, split={self.split}"
        )
        self.parquet_index = ParquetIndexWithMetadata(
            files=self.parquet_files,
            features=self.features,
            httpfs=httpfs,
            max_arrow_data_in_memory=max_arrow_data_in_memory,
            metadata_dir=self.parquet_metadata_directory,
        )

    def _init_viewer_index(self, hf_token: Optional[str], data_store: Optional[str]) -> None:
        logging.info(f"Create libviewer.Dataset for dataset={self.dataset}, config={self.config}, split={self.split}")

        # construct the required parquet_files list for libviewer.Dataset
        files = []
        for f in self.parquet_files:
            files.append(
                {
                    "path": f"{f['config']}/{f['split']}/{f['filename']}",
                    "size": f["size"],
                    "num_rows": f["num_rows"],
                    "metadata_path": f["parquet_metadata_subpath"],
                }
            )

        self.viewer_index = lv.Dataset(
            name=self.dataset,
            files=files,
            revision=self.revision,
            hf_token=hf_token,
            data_store=data_store,
            metadata_store=f"file://{self.parquet_metadata_directory}"
        )

    # note that this cache size is global for the class, not per instance
    @lru_cache(maxsize=1)
    def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
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
        if self.use_libviewer_for_datasets is True or (
            isinstance(self.use_libviewer_for_datasets, set) and self.dataset in self.use_libviewer_for_datasets
        ):
            try:
                return self.query_libviewer_index(offset=offset, length=length)
            except Exception as e:
                # list files at the metadata directory for debugging
                files = [str(f) for f in Path(self.parquet_metadata_directory).rglob("*")]
                raise ValueError(
                    f"Error while querying libviewer.Dataset: {e} for dataset={self.dataset},"
                    f" config={self.config}, split={self.split}. "
                    f"Parquet files: {self.parquet_files}. "
                    f"Parquet index files: {self.viewer_index.files}. "
                    f"Parquet metadata files: {files} at {self.parquet_metadata_directory}"
                ) from e
        else:
            return self.query_parquet_index(offset=offset, length=length)

    def query_parquet_index(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        """Query the parquet files using ParquetIndexWithMetadata.

        This is the old implementation without libviewer doing row-group pruning using pyarrow.
        """
        logging.info(
            f"Query {type(self.parquet_index).__name__} for dataset={self.dataset}, config={self.config},"
            f" split={self.split}, offset={offset}, length={length}"
        )
        return self.parquet_index.query(offset=offset, length=length)

    def query_libviewer_index(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        """Query the parquet files using libviewer.

        This is the new implementation using libviewer doing row-group and page pruning.
        """
        if not _has_libviewer:
            raise ImportError("libviewer is not installed but is required for page pruning")
        logging.info(
            f"Query libviewer.Dataset for dataset={self.dataset}, config={self.config},"
            f" split={self.split}, offset={offset}, length={length}, with page pruning"
        )
        # IndexError is not ideal but .query() raises it for invalid offsets
        if offset < 0:
            raise IndexError("Offset must be non-negative")
        if length < 0:
            raise IndexError("Length must be non-negative")

        try:
            batches = self.viewer_index.sync_scan(offset=offset, limit=length, scan_size_limit=self.max_scan_size)
        except lv.DatasetError as e:
            if "Scan size limit exceeded" in str(e):
                raise TooBigRows(str(e)) from e
            else:
                raise

        table = pa.Table.from_batches(batches, schema=self.features.arrow_schema)

        # FIXME(kszucs): binary truncation is implemented but disabled for now
        return truncate_binary_columns(table, max_binary_length=-1, features=self.features)
