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


class PresidioEntity(TypedDict):
    text: str
    type: str
    row_idx: int
    column_name: str


class PresidioAllEntitiesCountResponse(TypedDict):
    scanned_columns: list[str]
    num_in_vehicle_registration_entities: int
    num_organization_entities: int
    num_sg_nric_fin_entities: int
    num_person_entities: int
    num_credit_card_entities: int
    num_medical_license_entities: int
    num_nrp_entities: int
    num_us_ssn_entities: int
    num_crypto_entities: int
    num_date_time_entities: int
    num_location_entities: int
    num_us_driver_license_entities: int
    num_phone_number_entities: int
    num_url_entities: int
    num_us_passport_entities: int
    num_age_entities: int
    num_au_acn_entities: int
    num_email_address_entities: int
    num_in_pan_entities: int
    num_ip_address_entities: int
    num_id_entities: int
    num_us_bank_number_entities: int
    num_in_aadhaar_entities: int
    num_us_itin_entities: int
    num_au_medicare_entities: int
    num_iban_code_entities: int
    num_au_tfn_entities: int
    num_uk_nhs_entities: int
    num_email_entities: int
    num_au_abn_entities: int
    num_rows_with_in_vehicle_registration_entities: int
    num_rows_with_organization_entities: int
    num_rows_with_sg_nric_fin_entities: int
    num_rows_with_person_entities: int
    num_rows_with_credit_card_entities: int
    num_rows_with_medical_license_entities: int
    num_rows_with_nrp_entities: int
    num_rows_with_us_ssn_entities: int
    num_rows_with_crypto_entities: int
    num_rows_with_date_time_entities: int
    num_rows_with_location_entities: int
    num_rows_with_us_driver_license_entities: int
    num_rows_with_phone_number_entities: int
    num_rows_with_url_entities: int
    num_rows_with_us_passport_entities: int
    num_rows_with_age_entities: int
    num_rows_with_au_acn_entities: int
    num_rows_with_email_address_entities: int
    num_rows_with_in_pan_entities: int
    num_rows_with_ip_address_entities: int
    num_rows_with_id_entities: int
    num_rows_with_us_bank_number_entities: int
    num_rows_with_in_aadhaar_entities: int
    num_rows_with_us_itin_entities: int
    num_rows_with_au_medicare_entities: int
    num_rows_with_iban_code_entities: int
    num_rows_with_au_tfn_entities: int
    num_rows_with_uk_nhs_entities: int
    num_rows_with_email_entities: int
    num_rows_with_au_abn_entities: int
    num_scanned_rows: int
    has_scanned_columns: bool
    full_scan: Union[bool, None]


class PresidioEntitiesScanResponse(PresidioAllEntitiesCountResponse):
    entities: list[PresidioEntity]


class PresidioEntitiesCountResponse(TypedDict):
    scanned_columns: list[str]
    num_rows_with_person_entities: int
    num_rows_with_phone_number_entities: int
    num_rows_with_email_address_entities: int
    num_rows_with_sensitive_pii: int
    num_scanned_rows: int
    has_scanned_columns: bool
    full_scan: Union[bool, None]


class ImageUrlColumnsResponse(TypedDict):
    columns: list[str]


class ConfigInfoResponse(TypedDict):
    dataset_info: dict[str, Any]
    partial: bool


class ConfigParquetAndInfoResponse(TypedDict):
    parquet_files: list[SplitHubFile]
    dataset_info: dict[str, Any]
    estimated_dataset_info: Optional[dict[str, Any]]
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
    estimated_num_rows: Optional[int]


class SplitSize(TypedDict):
    dataset: str
    config: str
    split: str
    num_bytes_parquet_files: int
    num_bytes_memory: int
    num_rows: int
    num_columns: int
    estimated_num_rows: Optional[int]


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


DatasetModality = Literal["image", "audio", "text", "video", "geospatial", "3d", "tabular", "timeseries"]


class DatasetModalitiesResponse(TypedDict):
    modalities: list[DatasetModality]


class DatasetHubCacheResponse(TypedDict):
    preview: bool
    viewer: bool
    partial: bool
    num_rows: Optional[int]
    num_rows_source: Optional[Literal["full-exact", "partial-exact", "full-estimated"]]
    tags: list[DatasetTag]
    libraries: list[DatasetLibrary]
    modalities: list[DatasetModality]
    formats: list[DatasetFormat]


class _Filetype(TypedDict):
    extension: str
    count: int


class Filetype(_Filetype, total=False):
    archived_in: str
    compressed_in: str


class DatasetFiletypesResponse(TypedDict):
    filetypes: list[Filetype]


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
    estimated_num_rows: Optional[int]


class DatasetSizeContent(TypedDict):
    dataset: DatasetSize
    configs: list[ConfigSize]
    splits: list[SplitSize]


class DatasetSizeResponse(TypedDict):
    size: DatasetSizeContent
    pending: list[PreviousJob]
    failed: list[PreviousJob]
    partial: bool
