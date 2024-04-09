# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from environs import Env

StorageProtocol = Literal["file", "s3"]

ASSETS_BASE_URL = "http://localhost/assets"
ASSETS_STORAGE_PROTOCOL = "file"
ASSETS_STORAGE_ROOT = "/storage/assets"


@dataclass(frozen=True)
class AssetsConfig:
    base_url: str = ASSETS_BASE_URL
    storage_protocol: str = ASSETS_STORAGE_PROTOCOL
    storage_root: str = ASSETS_STORAGE_ROOT

    @classmethod
    def from_env(cls) -> "AssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=ASSETS_BASE_URL),
                storage_protocol=env.str(name="STORAGE_PROTOCOL", default=ASSETS_STORAGE_PROTOCOL),
                storage_root=env.str(name="STORAGE_ROOT", default=ASSETS_STORAGE_ROOT),
            )


S3_ACCESS_KEY_ID = None
S3_SECRET_ACCESS_KEY = None
S3_REGION_NAME = "us-east-1"


@dataclass(frozen=True)
class S3Config:
    access_key_id: Optional[str] = S3_ACCESS_KEY_ID
    secret_access_key: Optional[str] = S3_SECRET_ACCESS_KEY
    region_name: str = S3_REGION_NAME

    @classmethod
    def from_env(cls) -> "S3Config":
        env = Env(expand_vars=True)
        with env.prefixed("S3_"):
            return cls(
                access_key_id=env.str(name="ACCESS_KEY_ID", default=S3_ACCESS_KEY_ID),
                secret_access_key=env.str(name="SECRET_ACCESS_KEY", default=S3_SECRET_ACCESS_KEY),
                region_name=env.str(name="REGION_NAME", default=S3_REGION_NAME),
            )


CACHED_ASSETS_BASE_URL = "http://localhost/cached-assets"
CACHED_ASSETS_STORAGE_PROTOCOL = "file"
CACHED_ASSETS_STORAGE_ROOT = "/storage/cached-assets"


@dataclass(frozen=True)
class CachedAssetsConfig:
    base_url: str = CACHED_ASSETS_BASE_URL
    storage_protocol: str = CACHED_ASSETS_STORAGE_PROTOCOL
    storage_root: str = CACHED_ASSETS_STORAGE_ROOT

    @classmethod
    def from_env(cls) -> "CachedAssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHED_ASSETS_"):
            return cls(
                base_url=env.str(name="BASE_URL", default=CACHED_ASSETS_BASE_URL),
                storage_protocol=env.str(name="STORAGE_PROTOCOL", default=CACHED_ASSETS_STORAGE_PROTOCOL),
                storage_root=env.str(name="STORAGE_ROOT", default=CACHED_ASSETS_STORAGE_ROOT),
            )


CLOUDFRONT_EXPIRATION_SECONDS = 60 * 60 * 24
CLOUDFRONT_KEY_PAIR_ID = None
CLOUDFRONT_PRIVATE_KEY = None


@dataclass(frozen=True)
class CloudFrontConfig:
    expiration_seconds: int = CLOUDFRONT_EXPIRATION_SECONDS
    key_pair_id: Optional[str] = CLOUDFRONT_KEY_PAIR_ID
    private_key: Optional[str] = CLOUDFRONT_PRIVATE_KEY

    @classmethod
    def from_env(cls) -> "CloudFrontConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CLOUDFRONT_"):
            return cls(
                expiration_seconds=env.int(name="EXPIRATION_SECONDS", default=CLOUDFRONT_EXPIRATION_SECONDS),
                key_pair_id=env.str(name="KEY_PAIR_ID", default=CLOUDFRONT_KEY_PAIR_ID),
                private_key=env.str(name="PRIVATE_KEY", default=CLOUDFRONT_PRIVATE_KEY),
            )


PARQUET_METADATA_STORAGE_DIRECTORY = None


@dataclass(frozen=True)
class ParquetMetadataConfig:
    storage_directory: Optional[str] = PARQUET_METADATA_STORAGE_DIRECTORY

    @classmethod
    def from_env(cls) -> "ParquetMetadataConfig":
        env = Env(expand_vars=True)
        with env.prefixed("PARQUET_METADATA_"):
            return cls(
                storage_directory=env.str(name="STORAGE_DIRECTORY", default=PARQUET_METADATA_STORAGE_DIRECTORY),
            )


ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY = 300_000_000


@dataclass(frozen=True)
class RowsIndexConfig:
    max_arrow_data_in_memory: int = ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY

    @classmethod
    def from_env(cls) -> "RowsIndexConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ROWS_INDEX_"):
            return cls(
                max_arrow_data_in_memory=env.int(
                    name="MAX_ARROW_DATA_IN_MEMORY", default=ROWS_INDEX_MAX_ARROW_DATA_IN_MEMORY
                ),
            )


COMMON_BLOCKED_DATASETS: list[str] = []
COMMON_DATASET_SCRIPTS_ALLOW_LIST: list[str] = []
COMMON_HF_ENDPOINT = "https://huggingface.co"
COMMON_HF_TOKEN = None


@dataclass(frozen=True)
class CommonConfig:
    blocked_datasets: list[str] = field(default_factory=COMMON_BLOCKED_DATASETS.copy)
    dataset_scripts_allow_list: list[str] = field(default_factory=COMMON_DATASET_SCRIPTS_ALLOW_LIST.copy)
    hf_endpoint: str = COMMON_HF_ENDPOINT
    hf_token: Optional[str] = COMMON_HF_TOKEN

    @classmethod
    def from_env(cls) -> "CommonConfig":
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            return cls(
                blocked_datasets=env.list(name="BLOCKED_DATASETS", default=COMMON_BLOCKED_DATASETS.copy()),
                dataset_scripts_allow_list=env.list(
                    name="DATASET_SCRIPTS_ALLOW_LIST", default=COMMON_DATASET_SCRIPTS_ALLOW_LIST.copy()
                ),
                hf_endpoint=env.str(name="HF_ENDPOINT", default=COMMON_HF_ENDPOINT),
                hf_token=env.str(name="HF_TOKEN", default=COMMON_HF_TOKEN),  # nosec
            )


LOG_LEVEL = logging.INFO


@dataclass(frozen=True)
class LogConfig:
    level: int = LOG_LEVEL

    @classmethod
    def from_env(cls) -> "LogConfig":
        env = Env(expand_vars=True)
        with env.prefixed("LOG_"):
            return cls(
                level=env.log_level(name="LEVEL", default=LOG_LEVEL),
            )


CACHE_MONGO_DATABASE = "dataset_viewer_cache"
CACHE_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class CacheConfig:
    mongo_database: str = CACHE_MONGO_DATABASE
    mongo_url: str = CACHE_MONGO_URL

    @classmethod
    def from_env(cls) -> "CacheConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            return cls(
                mongo_database=env.str(name="MONGO_DATABASE", default=CACHE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=CACHE_MONGO_URL),
            )


QUEUE_MONGO_DATABASE = "dataset_viewer_queue"
QUEUE_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class QueueConfig:
    mongo_database: str = QUEUE_MONGO_DATABASE
    mongo_url: str = QUEUE_MONGO_URL

    @classmethod
    def from_env(cls) -> "QueueConfig":
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            return cls(
                mongo_database=env.str(name="MONGO_DATABASE", default=QUEUE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=QUEUE_MONGO_URL),
            )
