# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CommonConfig,
    LogConfig,
    MetricsConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

ADMIN_UVICORN_HOSTNAME = "localhost"
ADMIN_UVICORN_NUM_WORKERS = 2
ADMIN_UVICORN_PORT = 8000


@dataclass(frozen=True)
class UvicornConfig:
    hostname: str = ADMIN_UVICORN_HOSTNAME
    num_workers: int = ADMIN_UVICORN_NUM_WORKERS
    port: int = ADMIN_UVICORN_PORT

    @classmethod
    def from_env(cls) -> "UvicornConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ADMIN_UVICORN_"):
            return cls(
                hostname=env.str(name="HOSTNAME", default=ADMIN_UVICORN_HOSTNAME),
                num_workers=env.int(name="NUM_WORKERS", default=ADMIN_UVICORN_NUM_WORKERS),
                port=env.int(name="PORT", default=ADMIN_UVICORN_PORT),
            )


ADMIN_CACHE_REPORTS_NUM_RESULTS = 100
ADMIN_CACHE_REPORTS_WITH_CONTENT_NUM_RESULTS = 100
ADMIN_EXTERNAL_AUTH_URL = None
ADMIN_HF_ORGANIZATION = None
ADMIN_HF_TIMEOUT_SECONDS = 0.2
ADMIN_HF_WHOAMI_PATH = "/api/whoami-v2"
ADMIN_MAX_AGE = 10


@dataclass(frozen=True)
class AdminConfig:
    cache_reports_num_results: int = ADMIN_CACHE_REPORTS_NUM_RESULTS
    cache_reports_with_content_num_results: int = ADMIN_CACHE_REPORTS_WITH_CONTENT_NUM_RESULTS
    external_auth_url: Optional[str] = ADMIN_EXTERNAL_AUTH_URL  # not documented
    hf_organization: Optional[str] = ADMIN_HF_ORGANIZATION
    hf_timeout_seconds: Optional[float] = ADMIN_HF_TIMEOUT_SECONDS
    hf_whoami_path: str = ADMIN_HF_WHOAMI_PATH
    max_age: int = ADMIN_MAX_AGE

    @classmethod
    def from_env(cls, common_config: CommonConfig) -> "AdminConfig":
        env = Env(expand_vars=True)
        with env.prefixed("ADMIN_"):
            hf_whoami_path = env.str(name="HF_WHOAMI_PATH", default=ADMIN_HF_WHOAMI_PATH)
            external_auth_url = None if hf_whoami_path is None else f"{common_config.hf_endpoint}{hf_whoami_path}"
            return cls(
                cache_reports_num_results=env.int(
                    name="CACHE_REPORTS_NUM_RESULTS", default=ADMIN_CACHE_REPORTS_NUM_RESULTS
                ),
                cache_reports_with_content_num_results=env.int(
                    name="CACHE_REPORTS_WITH_CONTENT_NUM_RESULTS", default=ADMIN_CACHE_REPORTS_WITH_CONTENT_NUM_RESULTS
                ),
                external_auth_url=external_auth_url,
                hf_organization=env.str(name="HF_ORGANIZATION", default=ADMIN_HF_ORGANIZATION),
                hf_timeout_seconds=env.float(name="HF_TIMEOUT_SECONDS", default=ADMIN_HF_TIMEOUT_SECONDS),
                hf_whoami_path=hf_whoami_path,
                max_age=env.int(name="MAX_AGE", default=ADMIN_MAX_AGE),
            )


@dataclass(frozen=True)
class AppConfig:
    admin: AdminConfig = field(default_factory=AdminConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            assets=AssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            log=LogConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            admin=AdminConfig.from_env(common_config),
            metrics=MetricsConfig.from_env(),
        )
