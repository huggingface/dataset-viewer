# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass
from typing import Optional

from environs import Env

API_UVICORN_HOSTNAME = "localhost"
API_UVICORN_NUM_WORKERS = 2
API_UVICORN_PORT = 8000


@dataclass(frozen=True)
class UvicornConfig:
    hostname: str = API_UVICORN_HOSTNAME
    num_workers: int = API_UVICORN_NUM_WORKERS
    port: int = API_UVICORN_PORT

    @classmethod
    def from_env(cls) -> "UvicornConfig":
        env = Env(expand_vars=True)
        with env.prefixed("API_UVICORN_"):
            return cls(
                hostname=env.str(name="HOSTNAME", default=API_UVICORN_HOSTNAME),
                num_workers=env.int(name="NUM_WORKERS", default=API_UVICORN_NUM_WORKERS),
                port=env.int(name="PORT", default=API_UVICORN_PORT),
            )


API_EXTERNAL_AUTH_URL = None
API_HF_AUTH_PATH = "/api/datasets/%s/auth-check"
API_HF_JWT_PUBLIC_KEY_URL = None
API_HF_JWT_ALGORITHM = "EdDSA"
API_HF_TIMEOUT_SECONDS = 0.2
API_HF_WEBHOOK_SECRET = None
API_MAX_AGE_LONG = 120  # 2 minutes
API_MAX_AGE_SHORT = 10  # 10 seconds


@dataclass(frozen=True)
class ApiConfig:
    external_auth_url: Optional[str] = API_EXTERNAL_AUTH_URL  # not documented
    hf_auth_path: str = API_HF_AUTH_PATH
    hf_jwt_public_key_url: Optional[str] = API_HF_JWT_PUBLIC_KEY_URL
    hf_jwt_algorithm: Optional[str] = API_HF_JWT_ALGORITHM
    hf_timeout_seconds: Optional[float] = API_HF_TIMEOUT_SECONDS
    hf_webhook_secret: Optional[str] = API_HF_WEBHOOK_SECRET
    max_age_long: int = API_MAX_AGE_LONG
    max_age_short: int = API_MAX_AGE_SHORT

    @classmethod
    def from_env(cls, hf_endpoint: str) -> "ApiConfig":
        env = Env(expand_vars=True)
        with env.prefixed("API_"):
            hf_auth_path = env.str(name="HF_AUTH_PATH", default=API_HF_AUTH_PATH)
            external_auth_url = None if hf_auth_path is None else f"{hf_endpoint}{hf_auth_path}"
            return cls(
                external_auth_url=external_auth_url,
                hf_auth_path=hf_auth_path,
                hf_jwt_public_key_url=env.str(name="HF_JWT_PUBLIC_KEY_URL", default=API_HF_JWT_PUBLIC_KEY_URL),
                hf_jwt_algorithm=env.str(name="HF_JWT_ALGORITHM", default=API_HF_JWT_ALGORITHM),
                hf_timeout_seconds=env.float(name="HF_TIMEOUT_SECONDS", default=API_HF_TIMEOUT_SECONDS),
                hf_webhook_secret=env.str(name="HF_WEBHOOK_SECRET", default=API_HF_WEBHOOK_SECRET),
                max_age_long=env.int(name="MAX_AGE_LONG", default=API_MAX_AGE_LONG),
                max_age_short=env.int(name="MAX_AGE_SHORT", default=API_MAX_AGE_SHORT),
            )
