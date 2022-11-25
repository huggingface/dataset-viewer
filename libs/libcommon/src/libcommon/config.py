# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from environs import Env

from libcommon.asset import init_assets_dir
from libcommon.log import init_logging
from libcommon.queue import connect_to_queue_database
from libcommon.simple_cache import connect_to_cache_database


class CommonConfig:
    assets_base_url: str
    hf_endpoint: str
    hf_token: Optional[str]
    log_level: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("COMMON_"):
            self.assets_base_url = env.str(name="ASSETS_BASE_URL", default="assets")
            self.hf_endpoint = env.str(name="HF_ENDPOINT", default="https://huggingface.co")
            self.log_level = env.log_level(name="LOG_LEVEL", default="INFO")
            hf_token = env.str(name="HF_TOKEN", default="")
            self.hf_token = None if hf_token == "" else hf_token  # nosec
        self.setup()

    def setup(self):
        init_logging(self.log_level)


class CacheConfig:
    _assets_directory: Optional[str]
    assets_directory: str
    mongo_database: str
    mongo_url: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("CACHE_"):
            self._assets_directory = env.str(name="ASSETS_DIRECTORY", default=None)
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_cache")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
        self.setup()

    def setup(self):
        connect_to_cache_database(database=self.mongo_database, host=self.mongo_url)
        self.assets_directory = init_assets_dir(assets_directory=self._assets_directory)


class QueueConfig:
    max_jobs_per_namespace: int
    max_load_pct: int
    max_memory_pct: int
    mongo_database: str
    mongo_url: str
    sleep_seconds: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("QUEUE_"):
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_queue")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
            self.max_jobs_per_namespace = env.int(name="MAX_JOBS_PER_NAMESPACE", default=1)
            self.max_load_pct = env.int(name="MAX_LOAD_PCT", default=70)
            self.max_memory_pct = env.int(name="MAX_MEMORY_PCT", default=80)
            self.sleep_seconds = env.int(name="SLEEP_SECONDS", default=15)
        self.setup()

    def setup(self):
        connect_to_queue_database(database=self.mongo_database, host=self.mongo_url)
