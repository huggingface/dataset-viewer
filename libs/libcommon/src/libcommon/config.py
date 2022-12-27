# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


import logging
from dataclasses import dataclass, field
from typing import List, Optional

from environs import Env

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph, ProcessingGraphSpecification
from libcommon.queue import connect_to_queue_database
from libcommon.simple_cache import connect_to_cache_database
from libcommon.storage import init_dir

ASSETS_BASE_URL = "assets"
ASSETS_STORE_DIRECTORY = None


@dataclass
class AssetsConfig:
    base_url: str = ASSETS_BASE_URL
    _storage_directory: Optional[str] = ASSETS_STORE_DIRECTORY

    def __post_init__(self):
        self.storage_directory = init_dir(directory=self._storage_directory, appname="datasets_server_assets")

    @staticmethod
    def from_env() -> "AssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            return AssetsConfig(
                base_url=env.str(name="BASE_URL", default=ASSETS_BASE_URL),
                _storage_directory=env.str(name="STORAGE_DIRECTORY", default=ASSETS_STORE_DIRECTORY),
            )


COMMON_HF_ENDPOINT = "https://huggingface.co"
COMMON_HF_TOKEN = None
COMMON_LOG_LEVEL = logging.INFO


@dataclass
class CommonConfig:
    hf_endpoint: str = COMMON_HF_ENDPOINT
    hf_token: Optional[str] = COMMON_HF_TOKEN
    log_level: int = COMMON_LOG_LEVEL

    def __post_init__(self):
        init_logging(self.log_level)

    @staticmethod
    def from_env() -> "CommonConfig":
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            return CommonConfig(
                hf_endpoint=env.str(name="HF_ENDPOINT", default=COMMON_HF_ENDPOINT),
                hf_token=env.str(name="HF_TOKEN", default=COMMON_HF_TOKEN),  # nosec
                log_level=env.log_level(name="LOG_LEVEL", default=COMMON_LOG_LEVEL),
            )


CACHE_MONGO_DATABASE = "datasets_server_cache"
CACHE_MONGO_URL = "mongodb://localhost:27017"


@dataclass
class CacheConfig:
    mongo_database: str = CACHE_MONGO_DATABASE
    mongo_url: str = CACHE_MONGO_URL

    def __post_init__(self):
        connect_to_cache_database(database=self.mongo_database, host=self.mongo_url)

    @staticmethod
    def from_env() -> "CacheConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            return CacheConfig(
                mongo_database=env.str(name="MONGO_DATABASE", default=CACHE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=CACHE_MONGO_URL),
            )


QUEUE_MAX_JOBS_PER_NAMESPACE = 1
QUEUE_MONGO_DATABASE = "datasets_server_queue"
QUEUE_MONGO_URL = "mongodb://localhost:27017"


@dataclass
class QueueConfig:
    max_jobs_per_namespace: int = QUEUE_MAX_JOBS_PER_NAMESPACE
    mongo_database: str = QUEUE_MONGO_DATABASE
    mongo_url: str = QUEUE_MONGO_URL

    def __post_init__(self):
        connect_to_queue_database(database=self.mongo_database, host=self.mongo_url)

    @staticmethod
    def from_env() -> "QueueConfig":
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            return QueueConfig(
                max_jobs_per_namespace=env.int(name="MAX_JOBS_PER_NAMESPACE", default=QUEUE_MAX_JOBS_PER_NAMESPACE),
                mongo_database=env.str(name="MONGO_DATABASE", default=QUEUE_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=QUEUE_MONGO_URL),
            )


WORKER_LOOP_MAX_DISK_USAGE_PCT = 90
WORKER_LOOP_MAX_LOAD_PCT = 70
WORKER_LOOP_MAX_MEMORY_PCT = 80
WORKER_LOOP_SLEEP_SECONDS = 15


def get_empty_str_list() -> List[str]:
    return []


@dataclass
class WorkerLoopConfig:
    max_disk_usage_pct: int = WORKER_LOOP_MAX_DISK_USAGE_PCT
    max_load_pct: int = WORKER_LOOP_MAX_LOAD_PCT
    max_memory_pct: int = WORKER_LOOP_MAX_MEMORY_PCT
    sleep_seconds: int = WORKER_LOOP_SLEEP_SECONDS
    storage_paths: List[str] = field(default_factory=get_empty_str_list)

    @staticmethod
    def from_env() -> "WorkerLoopConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_LOOP_"):
            return WorkerLoopConfig(
                max_disk_usage_pct=env.int(name="MAX_DISK_USAGE_PCT", default=WORKER_LOOP_MAX_DISK_USAGE_PCT),
                max_load_pct=env.int(name="MAX_LOAD_PCT", default=WORKER_LOOP_MAX_LOAD_PCT),
                max_memory_pct=env.int(name="MAX_MEMORY_PCT", default=WORKER_LOOP_MAX_MEMORY_PCT),
                sleep_seconds=env.int(name="SLEEP_SECONDS", default=WORKER_LOOP_SLEEP_SECONDS),
                storage_paths=env.list(name="STORAGE_PATHS", default=get_empty_str_list()),
            )


@dataclass
class ProcessingGraphConfig:
    specification: ProcessingGraphSpecification = field(
        default_factory=lambda: {
            "/splits": {"input_type": "dataset", "required_by_dataset_viewer": True},
            "/parquet": {"input_type": "dataset"},
            "/first-rows": {"input_type": "split", "requires": "/splits", "required_by_dataset_viewer": True},
            "/size": {"input_type": "dataset", "requires": "/parquet"},
        }
    )

    def __post_init__(self):
        self.graph = ProcessingGraph(self.specification)

    @staticmethod
    def from_env() -> "ProcessingGraphConfig":
        # TODO: allow passing the graph via env vars
        return ProcessingGraphConfig()
