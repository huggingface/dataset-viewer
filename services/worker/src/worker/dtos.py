# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, TypedDict


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


class SplitsList(TypedDict):
    splits: List[SplitItem]


class FailedConfigItem(ConfigItem):
    error: Mapping[str, Any]


class DatasetSplitNamesResponse(TypedDict):
    splits: List[SplitItem]
    pending: List[ConfigItem]
    failed: List[FailedConfigItem]


class PreviousJob(SplitItem):
    kind: str


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Mapping[str, Any]


class RowItem(TypedDict):
    row_idx: int
    row: Mapping[str, Any]
    truncated_cells: List[str]


class SplitFirstRowsResponse(SplitItem):
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
    full_scan: Optional[bool]


class OptInOutUrlsScanResponse(OptInOutUrlsCountResponse):
    opt_in_urls: List[OptUrl]
    opt_out_urls: List[OptUrl]


class ImageUrlColumnsResponse(TypedDict):
    columns: List[str]


Row = Mapping[str, Any]


class RowsContent(TypedDict):
    rows: List[Row]
    all_fetched: bool


class ConfigInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]


class ParquetFileItem(SplitItem):
    url: str
    filename: str
    size: int


class ConfigParquetAndInfoResponse(TypedDict):
    parquet_files: List[ParquetFileItem]
    dataset_info: Dict[str, Any]


class ParquetFileMetadataItem(SplitItem):
    url: str
    filename: str
    size: int
    num_rows: int
    parquet_metadata_subpath: str


class ConfigParquetMetadataResponse(TypedDict):
    parquet_files_metadata: List[ParquetFileMetadataItem]


class ConfigParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]


class ConfigSize(TypedDict):
    dataset: str
    config: str
    num_bytes_original_files: int
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int


class SplitSize(SplitItem):
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int


class ConfigSizeContent(TypedDict):
    config: ConfigSize
    splits: list[SplitSize]


class ConfigSizeResponse(TypedDict):
    size: ConfigSizeContent


class ConfigNameItem(TypedDict):
    dataset: str
    config: str


class DatasetConfigNamesResponse(TypedDict):
    config_names: List[ConfigNameItem]


class DatasetInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]
    pending: List[PreviousJob]
    failed: List[PreviousJob]


class DatasetIsValidResponse(TypedDict):
    valid: bool


class DatasetParquetResponse(TypedDict):
    parquet_files: List[ParquetFileItem]
    pending: list[PreviousJob]
    failed: list[PreviousJob]


class DatasetSize(TypedDict):
    dataset: str
    num_bytes_original_files: int
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
