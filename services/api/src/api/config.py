# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

from environs import Env
from libcache.config import CacheConfig
from libcommon.config import CommonConfig
from libqueue.config import QueueConfig


class UvicornConfig:
    hostname: str
    num_workers: int
    port: int

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("API_UVICORN_"):
            self.hostname = env.str(name="HOSTNAME", default="localhost")
            self.num_workers = env.int(name="NUM_WORKERS", default=2)
            self.port = env.int(name="PORT", default=8000)


class ApiConfig:
    external_auth_url: Optional[str]
    hf_auth_path: str
    max_age_long: int
    max_age_short: int
    prometheus_multiproc_dir: Optional[str]

    def __init__(self, hf_endpoint: str):
        env = Env(expand_vars=True)
        with env.prefixed("API_"):
            self.hf_auth_path = env.str(name="HF_AUTH_PATH", default="/api/datasets/%s/auth-check")
            self.max_age_long = env.int(name="MAX_AGE_LONG", default=120)  # 2 minutes
            self.max_age_short = env.int(name="MAX_AGE_SHORT", default=10)  # 10 seconds
            self.prometheus_multiproc_dir = env.str(name="PROMETHEUS_MULTIPROC_DIR", default=None)
            self.external_auth_url = None if self.hf_auth_path is None else f"{hf_endpoint}{self.hf_auth_path}"


class AppConfig:
    api: ApiConfig
    cache: CacheConfig
    common: CommonConfig
    queue: QueueConfig

    def __init__(self):
        self.cache = CacheConfig()
        self.common = CommonConfig()
        self.queue = QueueConfig()
        self.api = ApiConfig(hf_endpoint=self.common.hf_endpoint)
