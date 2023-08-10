# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypedDict, Union

from libcommon.utils import FeatureItem, Row, RowItem, SplitHubFile


class JobRunnerInfo(TypedDict):
    job_type: str
    job_runner_version: int


@dataclass
class JobResult:
    content: Mapping[str, Any]
    progress: float

    def __post_init__(self) -> None:
        if self.progress < 0.0 or self.progress > 1.0:
            raise ValueError(f"Progress should be between 0 and 1, but got {self.progress}")


@dataclass
class CompleteJobResult(JobResult):
    content: Mapping[str, Any]
    progress: float = field(init=False, default=1.0)


class DatasetItem(TypedDict):
    dataset: str


class ConfigItem(DatasetItem):
    config: Optional[str]


class SplitItem(ConfigItem):
    split: Optional[str]


class FullConfigItem(DatasetItem):
    config: str


class FullSplitItem(FullConfigItem):
    split: str


class SplitsList(TypedDict):
    splits: List[FullSplitItem]


class FailedConfigItem(FullConfigItem):
    error: Mapping[str, Any]


class DatasetSplitNamesResponse(TypedDict):
    splits: List[FullSplitItem]
    pending: List[FullConfigItem]
    failed: List[FailedConfigItem]


class PreviousJob(TypedDict):
    dataset: str
    config: Optional[str]
    split: Optional[Union[str, None]]
    kind: str


class SplitFirstRowsResponse(FullSplitItem):
    features: List[FeatureItem]
    rows: List[RowItem]


class OptUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


class OptInOutUrlsCountResponse(TypedDict):
    urls_columns: List[str]
    num_opt_in_urls: int
    num_opt_out_urls: int
    num_urls: int
    num_scanned_rows: int
    has_urls_columns: bool
    full_scan: Union[bool, None]


class OptInOutUrlsScanResponse(OptInOutUrlsCountResponse):
    opt_in_urls: List[OptUrl]
    opt_out_urls: List[OptUrl]


class ImageUrlColumnsResponse(TypedDict):
    columns: List[str]


class RowsContent(TypedDict):
    rows: List[Row]
    all_fetched: bool


class ConfigInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]
    partial: bool


class ConfigParquetAndInfoResponse(TypedDict):
    parquet_files: List[SplitHubFile]
    dataset_info: Dict[str, Any]
    partial: bool


class ParquetFileMetadataItem(SplitItem):
    url: str
    filename: str
    size: int
    num_rows: int
    parquet_metadata_subpath: str


class ConfigParquetMetadataResponse(TypedDict):
    parquet_files_metadata: List[ParquetFileMetadataItem]
    features: Optional[Dict[str, Any]]
    partial: bool


class ConfigParquetResponse(TypedDict):
    parquet_files: List[SplitHubFile]
    features: Optional[Dict[str, Any]]
    partial: bool


class ConfigSize(TypedDict):
    dataset: str
    config: str
    num_bytes_original_files: Optional[
        int
    ]  # optional because partial parquet conversion can't provide the size of the original data
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int


class SplitSize(TypedDict):
    dataset: str
    config: str
    split: str
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int


class ConfigSizeContent(TypedDict):
    config: ConfigSize
    splits: list[SplitSize]


class ConfigSizeResponse(TypedDict):
    size: ConfigSizeContent
    partial: bool


class ConfigNameItem(TypedDict):
    dataset: str
    config: str


class DatasetConfigNamesResponse(TypedDict):
    config_names: List[ConfigNameItem]


class DatasetInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]
    pending: List[PreviousJob]
    failed: List[PreviousJob]
    partial: bool


class IsValidResponse(TypedDict):
    preview: bool
    viewer: bool
    search: bool


class DatasetParquetResponse(TypedDict):
    parquet_files: List[SplitHubFile]
    pending: list[PreviousJob]
    failed: list[PreviousJob]
    partial: bool


class DatasetSize(TypedDict):
    dataset: str
    num_bytes_original_files: Optional[int]
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int


class DatasetSizeContent(TypedDict):
    dataset: DatasetSize
    configs: list[ConfigSize]
    splits: list[SplitSize]


class DatasetSizeResponse(TypedDict):
    size: DatasetSizeContent
    pending: list[PreviousJob]
    failed: list[PreviousJob]
    partial: bool
