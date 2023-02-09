# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field
from typing import Optional

from environs import Env
from libcommon.config import (
    CacheConfig,
    CommonConfig,
    ProcessingGraphConfig,
    QueueConfig,
)

API_UVICORN_HOSTNAME = "localhost"
API_UVICORN_NUM_WORKERS = 2
API_UVICORN_PORT = 8000


@dataclass
class UvicornConfig:
    hostname: str = API_UVICORN_HOSTNAME
    num_workers: int = API_UVICORN_NUM_WORKERS
    port: int = API_UVICORN_PORT

    @staticmethod
    def from_env() -> "UvicornConfig":
        env = Env(expand_vars=True)
        with env.prefixed("API_UVICORN_"):
            return UvicornConfig(
                hostname=env.str(name="HOSTNAME", default=API_UVICORN_HOSTNAME),
                num_workers=env.int(name="NUM_WORKERS", default=API_UVICORN_NUM_WORKERS),
                port=env.int(name="PORT", default=API_UVICORN_PORT),
            )


API_EXTERNAL_AUTH_URL = None
API_HF_AUTH_PATH = "/api/datasets/%s/auth-check"
API_MAX_AGE_LONG = 120  # 2 minutes
API_MAX_AGE_SHORT = 10  # 10 seconds


@dataclass
class ApiConfig:
    external_auth_url: Optional[str] = API_EXTERNAL_AUTH_URL  # not documented
    hf_auth_path: str = API_HF_AUTH_PATH
    max_age_long: int = API_MAX_AGE_LONG
    max_age_short: int = API_MAX_AGE_SHORT

    @staticmethod
    def from_env(common_config: CommonConfig) -> "ApiConfig":
        env = Env(expand_vars=True)
        with env.prefixed("API_"):
            hf_auth_path = env.str(name="HF_AUTH_PATH", default=API_HF_AUTH_PATH)
            external_auth_url = None if hf_auth_path is None else f"{common_config.hf_endpoint}{hf_auth_path}"
            return ApiConfig(
                external_auth_url=external_auth_url,
                hf_auth_path=hf_auth_path,
                max_age_long=env.int(name="MAX_AGE_LONG", default=API_MAX_AGE_LONG),
                max_age_short=env.int(name="MAX_AGE_SHORT", default=API_MAX_AGE_SHORT),
            )


@dataclass
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    processing_graph: ProcessingGraphConfig = field(default_factory=ProcessingGraphConfig)

    @staticmethod
    def from_env() -> "AppConfig":
        common_config = CommonConfig.from_env()
        return AppConfig(
            common=common_config,
            cache=CacheConfig.from_env(),
            processing_graph=ProcessingGraphConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(common_config=common_config),
        )
