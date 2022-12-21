# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from environs import Env

from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph, ProcessingGraphSpecification
from libcommon.queue import connect_to_queue_database
from libcommon.simple_cache import connect_to_cache_database
from libcommon.storage import init_dir


class AssetsConfig:
    base_url: str
    storage_directory: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("ASSETS_"):
            self.base_url = env.str(name="BASE_URL", default="assets")
            self._storage_directory = env.str(name="STORAGE_DIRECTORY", default=None)
        self.setup()

    def setup(self):
        self.storage_directory = init_dir(directory=self._storage_directory, appname="datasets_server_assets")


class CommonConfig:
    hf_endpoint: str
    hf_token: Optional[str]
    log_level: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            self.hf_endpoint = env.str(name="HF_ENDPOINT", default="https://huggingface.co")
            self.log_level = env.log_level(name="LOG_LEVEL", default="INFO")
            hf_token = env.str(name="HF_TOKEN", default="")
            self.hf_token = None if hf_token == "" else hf_token  # nosec
        self.setup()

    def setup(self):
        init_logging(self.log_level)


class CacheConfig:
    mongo_database: str
    mongo_url: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_cache")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
        self.setup()

    def setup(self):
        connect_to_cache_database(database=self.mongo_database, host=self.mongo_url)


class QueueConfig:
    max_jobs_per_namespace: int
    mongo_database: str
    mongo_url: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_queue")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
            self.max_jobs_per_namespace = env.int(name="MAX_JOBS_PER_NAMESPACE", default=1)
        self.setup()

    def setup(self):
        connect_to_queue_database(database=self.mongo_database, host=self.mongo_url)


class WorkerLoopConfig:
    max_load_pct: int
    max_memory_pct: int
    sleep_seconds: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("WORKER_LOOP_"):
            self.max_load_pct = env.int(name="MAX_LOAD_PCT", default=70)
            self.max_memory_pct = env.int(name="MAX_MEMORY_PCT", default=80)
            self.sleep_seconds = env.int(name="SLEEP_SECONDS", default=15)


class ProcessingGraphConfig:
    specification: ProcessingGraphSpecification
    graph: ProcessingGraph

    def __init__(self):
        # TODO: allow passing the graph via env vars
        self.specification = {
            "/splits": {"input_type": "dataset", "required_by_dataset_viewer": True},
            "/parquet": {"input_type": "dataset"},
            "/first-rows": {"input_type": "split", "requires": "/splits", "required_by_dataset_viewer": True},
        }

        self.setup()

    def setup(self):
        self.graph = ProcessingGraph(self.specification)
