# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict, Union

from libcommon.dtos import FullConfigItem, FullSplitItem, SplitHubFile, SplitItem


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


class SplitsList(TypedDict):
    splits: list[FullSplitItem]


class FailedConfigItem(FullConfigItem):
    error: Mapping[str, Any]


class DatasetSplitNamesResponse(TypedDict):
    splits: list[FullSplitItem]
    pending: list[FullConfigItem]
    failed: list[FailedConfigItem]


class PreviousJob(TypedDict):
    dataset: str
    config: Optional[str]
    split: Optional[Union[str, None]]
    kind: str


class OptUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


class OptInOutUrlsCountResponse(TypedDict):
    urls_columns: list[str]
    num_opt_in_urls: int
    num_opt_out_urls: int
    num_urls: int
    num_scanned_rows: int
    has_urls_columns: bool
    full_scan: Union[bool, None]


class OptInOutUrlsScanResponse(OptInOutUrlsCountResponse):
    opt_in_urls: list[OptUrl]
    opt_out_urls: list[OptUrl]


class ImageUrlColumnsResponse(TypedDict):
    columns: list[str]


class ConfigInfoResponse(TypedDict):
    dataset_info: dict[str, Any]
    partial: bool


class ConfigParquetAndInfoResponse(TypedDict):
    parquet_files: list[SplitHubFile]
    dataset_info: dict[str, Any]
    partial: bool


class ParquetFileMetadataItem(SplitItem):
    url: str
    filename: str
    size: int
    num_rows: int
    parquet_metadata_subpath: str


class ConfigParquetMetadataResponse(TypedDict):
    parquet_files_metadata: list[ParquetFileMetadataItem]
    features: Optional[dict[str, Any]]
    partial: bool


class ConfigParquetResponse(TypedDict):
    parquet_files: list[SplitHubFile]
    features: Optional[dict[str, Any]]
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


class SplitDuckdbIndex(SplitHubFile):
    features: Optional[dict[str, Any]]
    has_fts: bool
    # The following fields can be None in old cache entries
    partial: Optional[bool]
    num_rows: Optional[int]
    num_bytes: Optional[int]
    duckdb_version: str


class SplitDuckdbIndexSize(TypedDict):
    dataset: str
    config: str
    split: str
    has_fts: bool
    num_rows: int
    num_bytes: int


class ConfigDuckdbIndexSize(TypedDict):
    dataset: str
    config: str
    has_fts: bool
    num_rows: int
    num_bytes: int


class ConfigDuckdbIndexSizeContent(TypedDict):
    config: ConfigDuckdbIndexSize
    splits: list[SplitDuckdbIndexSize]


class ConfigDuckdbIndexSizeResponse(TypedDict):
    size: ConfigDuckdbIndexSizeContent
    partial: bool


class DatasetDuckdbIndexSize(TypedDict):
    dataset: str
    has_fts: bool
    num_rows: int
    num_bytes: int


class DatasetDuckdbIndexSizeContent(TypedDict):
    dataset: DatasetDuckdbIndexSize
    configs: list[ConfigDuckdbIndexSize]
    splits: list[SplitDuckdbIndexSize]


class DatasetDuckdbIndexSizeResponse(TypedDict):
    size: DatasetDuckdbIndexSizeContent
    pending: list[PreviousJob]
    failed: list[PreviousJob]
    partial: bool


class ConfigNameItem(TypedDict):
    dataset: str
    config: str


class DatasetConfigNamesResponse(TypedDict):
    config_names: list[ConfigNameItem]


class DatasetInfoResponse(TypedDict):
    dataset_info: dict[str, Any]
    pending: list[PreviousJob]
    failed: list[PreviousJob]
    partial: bool


class IsValidResponse(TypedDict):
    preview: bool
    viewer: bool
    search: bool
    filter: bool
    statistics: bool


DatasetTag = Literal["croissant"]
DatasetLibrary = Literal["mlcroissant", "webdataset", "datasets", "pandas", "dask"]
DatasetFormat = Literal["json", "csv", "parquet", "imagefolder", "audiofolder", "webdataset", "text"]
ProgrammingLanguage = Literal["python"]


class LoadingCode(TypedDict):
    config_name: str
    arguments: dict[str, Any]
    code: str


class CompatibleLibrary(TypedDict):
    language: ProgrammingLanguage
    library: DatasetLibrary
    function: str
    loading_codes: list[LoadingCode]


class DatasetCompatibleLibrariesResponse(TypedDict):
    tags: list[DatasetTag]
    libraries: list[CompatibleLibrary]
    formats: list[DatasetFormat]


DatasetModality = Literal["image", "audio", "text"]


class DatasetModalitiesResponse(TypedDict):
    modalities: list[DatasetModality]


class DatasetHubCacheResponse(TypedDict):
    preview: bool
    viewer: bool
    partial: bool
    num_rows: int
    tags: list[DatasetTag]
    libraries: list[DatasetLibrary]
    modalities: list[DatasetModality]
    formats: list[DatasetFormat]


class DatasetParquetResponse(TypedDict):
    parquet_files: list[SplitHubFile]
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
