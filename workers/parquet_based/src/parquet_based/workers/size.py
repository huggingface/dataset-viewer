# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import asdict, dataclass
from typing import Any, List, Mapping, TypedDict

import hffs

# https://github.com/apache/arrow/issues/32609
import pyarrow.parquet as pq  # type: ignore

from parquet_based.workers._parquet_based_worker import ParquetBasedWorker, ParquetFile


@dataclass
class ParquetFileWithSize(ParquetFile):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int
    num_rows: int = 0
    num_columns: int = 0
    serialized_size: int = 0

    def compute_size(self) -> "ParquetFileWithSize":
        REVISION = "refs/convert/parquet"
        fs = hffs.HfFileSystem(self.dataset, repo_type="dataset", revision=REVISION)
        metadata = pq.read_metadata(f"{self.config}/{self.filename}", filesystem=fs)
        # ^ are we streaming to only read the metadata in the footer, or is the whole parquet file read?
        self.num_rows = metadata.num_rows
        self.num_columns = metadata.num_columns
        self.serialized_size = metadata.serialized_size
        return self

    def as_item(self) -> "ParquetFileItem":
        return {
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "url": self.url,
            "filename": self.filename,
            "num_bytes": self.size,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "serialized_size": self.serialized_size,
            "size": self.size,
        }


class ParquetFileItem(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    num_bytes: int
    num_rows: int
    num_columns: int
    serialized_size: int
    size: int


class SplitItem(TypedDict):
    dataset: str
    config: str
    split: str
    num_parquet_files: int
    num_bytes: int
    num_rows: int
    num_columns: int


class ConfigItem(TypedDict):
    dataset: str
    config: str
    num_splits: int
    num_parquet_files: int
    num_bytes: int
    num_rows: int
    # num_columns: int


class DatasetItem(TypedDict):
    dataset: str
    num_configs: int
    num_splits: int
    num_parquet_files: int
    num_bytes: int
    num_rows: int
    # num_columns: int


class SizeResponseContent(TypedDict):
    # dataset: DatasetItem
    # configs: List[ConfigItem]
    # splits: List[SplitItem]
    parquet_files: List[ParquetFileItem]


def get_size_response_content(parquet_files: List[ParquetFile]) -> SizeResponseContent:
    # get the number of sample of each parquet file, and compute the response
    parquet_files_with_size: List[ParquetFileWithSize] = [
        ParquetFileWithSize(**(asdict(parquet_file))).compute_size() for parquet_file in parquet_files
    ]
    # return the list of splits parquet files with their size and number of samples
    return {"parquet_files": [parquet_file_with_size.as_item() for parquet_file_with_size in parquet_files_with_size]}


class SizeWorker(ParquetBasedWorker):
    @staticmethod
    def get_job_type() -> str:
        return "/size"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def compute(self) -> Mapping[str, Any]:
        return get_size_response_content(parquet_files=self.parquet_files)
