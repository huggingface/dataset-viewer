import logging
import os
from pathlib import Path
from typing import Optional, TypedDict
from urllib.parse import unquote

import libviewer as lv
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from async_lru import alru_cache
from datasets import Features, Value
from datasets.table import CastError, cast_table_to_schema

from libcommon.constants import CONFIG_PARQUET_METADATA_KIND
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath

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


class RowsIndex:
    def __init__(
        self,
        dataset: str,
        config: str,
        split: str,
        parquet_metadata_directory: StrPath,
        max_scan_size: Optional[int] = None,
        hf_token: Optional[str] = None,
        hf_endpoint: Optional[str] = None,
        data_store: Optional[str] = None,
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.max_scan_size = max_scan_size
        self.parquet_metadata_directory = Path(parquet_metadata_directory)

        self._init_dataset_info()
        self._init_viewer_index(hf_token=hf_token, hf_endpoint=hf_endpoint, data_store=data_store)

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

        # calculate the total number of rows, required for the `first_rows` response
        self.num_rows_total = sum(f["num_rows"] for f in self.parquet_files)
        # required for `rows` endpoint's response
        self.partial = parquet_export_is_partial(self.parquet_files[0]["url"])

        # retrieve the features from the mongo response
        features = response["content"].get("features")
        if features:
            self.features = Features.from_dict(features)
        else:
            # config-parquet version<6 didn't have features
            first_metadata_file = self.parquet_metadata_directory / parquet_files[0]["parquet_metadata_subpath"]
            arrow_schema = pq.read_schema(first_metadata_file)
            self.features = Features.from_arrow_schema(arrow_schema)

    def _init_viewer_index(
        self, hf_token: Optional[str], hf_endpoint: Optional[str], data_store: Optional[str]
    ) -> None:
        logging.info(f"Create libviewer.Dataset for dataset={self.dataset}, config={self.config}, split={self.split}")

        # construct the required parquet_files list for libviewer.Dataset

        files = [
            {
                "path": f["url"].split("/resolve/", 1)[1].split("/", 1)[1],
                "size": f["size"],
                "num_rows": f["num_rows"],
                "metadata_path": f["parquet_metadata_subpath"],
            }
            for f in self.parquet_files
        ]
        revision = unquote(self.parquet_files[0]["url"].split("/resolve/", 1)[1].split("/", 1)[0])

        self.viewer_index = lv.Dataset(
            name=self.dataset,
            files=files,
            revision=revision,
            hf_token=hf_token,
            hf_endpoint=hf_endpoint,
            data_store=data_store,
            metadata_store=f"file://{self.parquet_metadata_directory}",
        )

    # note that this cache size is global for the class, not per instance
    @alru_cache(maxsize=1)
    async def query(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
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
        return await self.query_libviewer_index(offset=offset, length=length)

    async def query_libviewer_index(self, offset: int, length: int) -> tuple[pa.Table, list[str]]:
        """Query the parquet files using libviewer.

        This is the new implementation using libviewer doing row-group and page pruning.
        """
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
            batches, _files_to_index = await self.viewer_index.scan(
                offset=offset, limit=length, scan_size_limit=self.max_scan_size
            )
        except lv.DatasetError as e:
            if "Scan size limit exceeded" in str(e):
                raise TooBigRows(
                    str(e)
                    + f"\n\nMake sure that\n\n1. individual rows are of reasonable size (max {size_str(self.max_scan_size // length)}/row)\n2. the Parquet files contain a page index to enable random access without loading entire row groups"
                ) from e
            else:
                raise

        if len(batches) > 0:
            try:
                parts = [
                    cast_table_to_schema(pa.Table.from_batches([batch]), self.features.arrow_schema)
                    for batch in batches
                ]
            except CastError as e:
                raise SchemaMismatchError(
                    f"The schema of the parquet files does not match the expected schema: {e}"
                ) from e
            table = pa.concat_tables(parts)
        else:
            table = pa.Table.from_batches([], schema=self.features.arrow_schema)

        # FIXME(kszucs): binary truncation is implemented but disabled for now
        # since we can't iterate on small batches in sync_scan() and truncate batches per batches yet
        return truncate_binary_columns(table, max_binary_length=-1, features=self.features)
