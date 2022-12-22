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


@dataclass
class AssetsConfig:
    base_url: str = "assets"
    _storage_directory: Optional[str] = None

    def __post_init__(self):
        self.storage_directory = init_dir(directory=self._storage_directory, appname="datasets_server_assets")

    @staticmethod
    def from_env() -> "AssetsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            return AssetsConfig(
                base_url=env.str(name="BASE_URL", default=None),
                _storage_directory=env.str(name="STORAGE_DIRECTORY", default=None),
            )


@dataclass
class CommonConfig:
    hf_endpoint: str = "https://huggingface.co"
    hf_token: Optional[str] = None
    log_level: int = logging.INFO

    def __post_init__(self):
        init_logging(self.log_level)

    @staticmethod
    def from_env() -> "CommonConfig":
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            return CommonConfig(
                hf_endpoint=env.str(name="HF_ENDPOINT", default=None),
                log_level=env.log_level(name="LOG_LEVEL", default=None),
                hf_token=env.str(name="HF_TOKEN", default=None),  # nosec
            )


@dataclass
class CacheConfig:
    mongo_database: str = "datasets_server_cache"
    mongo_url: str = "mongodb://localhost:27017"

    def __post_init__(self):
        connect_to_cache_database(database=self.mongo_database, host=self.mongo_url)

    @staticmethod
    def from_env() -> "CacheConfig":
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            return CacheConfig(
                mongo_database=env.str(name="MONGO_DATABASE", default=None),
                mongo_url=env.str(name="MONGO_URL", default=None),
            )


@dataclass
class QueueConfig:
    max_jobs_per_namespace: int = 1
    mongo_database: str = "datasets_server_queue"
    mongo_url: str = "mongodb://localhost:27017"

    def __post_init__(self):
        connect_to_queue_database(database=self.mongo_database, host=self.mongo_url)

    @staticmethod
    def from_env() -> "QueueConfig":
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            return QueueConfig(
                max_jobs_per_namespace=env.int(name="MAX_JOBS_PER_NAMESPACE", default=None),
                mongo_database=env.str(name="MONGO_DATABASE", default=None),
                mongo_url=env.str(name="MONGO_URL", default=None),
            )


@dataclass
class WorkerLoopConfig:
    max_disk_usage_pct: int = 90
    max_load_pct: int = 70
    max_memory_pct: int = 80
    sleep_seconds: int = 15
    storage_paths: List[str] = field(default_factory=lambda: [])

    @staticmethod
    def from_env() -> "WorkerLoopConfig":
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_LOOP_"):
            return WorkerLoopConfig(
                max_disk_usage_pct=env.int(name="MAX_DISK_USAGE_PCT", default=None),
                max_load_pct=env.int(name="MAX_LOAD_PCT", default=None),
                max_memory_pct=env.int(name="MAX_MEMORY_PCT", default=None),
                sleep_seconds=env.int(name="SLEEP_SECONDS", default=None),
                storage_paths=env.list(name="STORAGE_PATHS", default=None),
            )


@dataclass
class ProcessingGraphConfig:
    specification: ProcessingGraphSpecification = field(
        default_factory=lambda: {
            "/splits": {"input_type": "dataset", "required_by_dataset_viewer": True},
            "/parquet": {"input_type": "dataset"},
            "/first-rows": {"input_type": "split", "requires": "/splits", "required_by_dataset_viewer": True},
        }
    )

    def __post_init__(self):
        self.graph = ProcessingGraph(self.specification)

    @staticmethod
    def from_env() -> "ProcessingGraphConfig":
        # TODO: allow passing the graph via env vars
        return ProcessingGraphConfig()
