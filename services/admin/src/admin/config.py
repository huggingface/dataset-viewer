# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

ADMIN_UVICORN_HOSTNAME = "localhost"
ADMIN_UVICORN_NUM_WORKERS = 2
ADMIN_UVICORN_PORT = 8000


@dataclass
class UvicornConfig:
    hostname: str = ADMIN_UVICORN_HOSTNAME
    num_workers: int = ADMIN_UVICORN_NUM_WORKERS
    port: int = ADMIN_UVICORN_PORT

    @staticmethod
    def from_env() -> "UvicornConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ADMIN_UVICORN_"):
            return UvicornConfig(
                hostname=env.str(name="HOSTNAME", default=ADMIN_UVICORN_HOSTNAME),
                num_workers=env.int(name="NUM_WORKERS", default=ADMIN_UVICORN_NUM_WORKERS),
                port=env.int(name="PORT", default=ADMIN_UVICORN_PORT),
            )


ADMIN_CACHE_REPORTS_NUM_RESULTS = 100
ADMIN_HF_ORGANIZATION = None
ADMIN_HF_WHOAMI_PATH = "/api/whoami-v2"
ADMIN_MAX_AGE = 10


@dataclass
class AdminConfig:
    cache_reports_num_results: int = ADMIN_CACHE_REPORTS_NUM_RESULTS
    hf_organization: Optional[str] = ADMIN_HF_ORGANIZATION
    hf_whoami_path: str = ADMIN_HF_WHOAMI_PATH
    max_age: int = ADMIN_MAX_AGE

    @staticmethod
    def from_env() -> "AdminConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ADMIN_"):
            return AdminConfig(
                cache_reports_num_results=env.int(
                    name="CACHE_REPORTS_NUM_RESULTS", default=ADMIN_CACHE_REPORTS_NUM_RESULTS
                ),
                hf_organization=env.str(name="HF_ORGANIZATION", default=ADMIN_HF_ORGANIZATION),
                hf_whoami_path=env.str(name="HF_WHOAMI_PATH", default=ADMIN_HF_WHOAMI_PATH),
                max_age=env.int(name="MAX_AGE", default=ADMIN_MAX_AGE),
            )


@dataclass
class AppConfig:
    admin: AdminConfig = field(default_factory=AdminConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)

    def __post_init__(self):
        self.external_auth_url = (
            None if self.admin.hf_whoami_path is None else f"{self.common.hf_endpoint}{self.admin.hf_whoami_path}"
        )

    @staticmethod
    def from_env() -> "AppConfig":
        # First process the common configuration to setup the logging
        return AppConfig(
            common=CommonConfig.from_env(),
            assets=AssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            admin=AdminConfig.from_env(),
        )
