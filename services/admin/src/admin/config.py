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
        with env.prefixed("ADMIN_UVICORN_"):
            self.hostname = env.str(name="HOSTNAME", default="localhost")
            self.num_workers = env.int(name="NUM_WORKERS", default=2)
            self.port = env.int(name="PORT", default=8000)


class AdminConfig:
    cache_reports_num_results: int
    external_auth_url: str
    hf_organization: Optional[str]
    hf_whoami_path: str
    max_age: int
    prometheus_multiproc_dir: Optional[str]

    def __init__(self, hf_endpoint: str):
        env = Env(expand_vars=True)
        with env.prefixed("ADMIN_"):
            self.hf_organization = env.str(name="HF_ORGANIZATION", default=None)
            self.cache_reports_num_results = env.int(name="CACHE_REPORTS_NUM_RESULTS", default=100)
            self.hf_whoami_path = env.str(name="HF_WHOAMI_PATH", default="/api/whoami-v2")
            self.max_age = env.int(name="MAX_AGE", default=10)  # 10 seconds
            self.prometheus_multiproc_dir = env.str(name="PROMETHEUS_MULTIPROC_DIR", default=None)
            self.external_auth_url = None if self.hf_whoami_path is None else f"{hf_endpoint}{self.hf_whoami_path}"


class AppConfig:
    admin: AdminConfig
    cache: CacheConfig
    common: CommonConfig
    queue: QueueConfig

    def __init__(self):
        self.cache = CacheConfig()
        self.common = CommonConfig()
        self.queue = QueueConfig()
        self.admin = AdminConfig(hf_endpoint=self.common.hf_endpoint)
