# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

"""Read secrets from files mounted into the pod, so they never have to be injected as environment
variables.

A CSI driver mounts each secret as its own file in a tmpfs directory. The application reads them once
at startup and keeps them in memory. They are deliberately never written back to ``os.environ``, which
would defeat the purpose by exposing them through ``/proc/<pid>/environ`` and to every subprocess.

Nothing here knows about Infisical. The contract is a directory of files, so the same code works behind
any CSI provider or injector.

When the provider bounds how long the secrets stay readable, a file is served empty once its window
closes, and a freshly granted window takes a few seconds to take effect. An empty file is therefore
treated as "not there yet" and retried, rather than as an absent secret.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import monotonic, sleep
from types import MappingProxyType
from typing import Optional, overload

from environs import Env

SECRETS_DIR = ""
SECRETS_TIMEOUT_SECONDS = 30.0
SECRETS_POLL_INTERVAL_SECONDS = 1.0


class SecretsError(RuntimeError):
    pass


@dataclass(frozen=True)
class SecretsConfig:
    directory: str = SECRETS_DIR
    timeout_seconds: float = SECRETS_TIMEOUT_SECONDS
    poll_interval_seconds: float = SECRETS_POLL_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "SecretsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("SECRETS_"):
            return cls(
                directory=env.str(name="DIR", default=SECRETS_DIR),
                timeout_seconds=env.float(name="TIMEOUT_SECONDS", default=SECRETS_TIMEOUT_SECONDS),
                poll_interval_seconds=env.float(name="POLL_INTERVAL_SECONDS", default=SECRETS_POLL_INTERVAL_SECONDS),
            )

    @property
    def enabled(self) -> bool:
        return bool(self.directory)


def _read_directory(directory: Path) -> dict[str, str]:
    """Return the secrets currently in the mount, keyed by file name.

    Entries starting with a dot are skipped: the atomic writer both the kubelet and the CSI driver use
    keeps its versioned copies in `..data` and `..<timestamp>` and symlinks the real names alongside
    them.
    """
    secrets: dict[str, str] = {}
    for entry in sorted(os.listdir(directory)):
        if entry.startswith("."):
            continue
        path = directory / entry
        if path.is_file():
            secrets[entry] = path.read_text().strip()
    return secrets


@lru_cache(maxsize=1)
def get_secrets() -> Mapping[str, str]:
    """Return the mounted secrets, read once per process.

    Empty when no directory is configured, which is the case for local development, the tests and the
    e2e suite: the config classes then keep reading their environment variables as before.
    """
    config = SecretsConfig.from_env()
    if not config.enabled:
        return MappingProxyType({})

    directory = Path(config.directory)
    if not directory.is_dir():
        raise SecretsError(f"the secrets directory {config.directory} is not mounted")

    deadline = monotonic() + config.timeout_seconds
    while True:
        secrets = _read_directory(directory)
        empty = sorted(name for name, value in secrets.items() if not value)
        if not empty:
            logging.info(f"read {len(secrets)} secrets from {config.directory} at startup")
            return MappingProxyType(secrets)
        if monotonic() >= deadline:
            raise SecretsError(
                f"still empty after {config.timeout_seconds:g}s in {config.directory}: {', '.join(empty)}."
                " The read window may have closed, or never been granted."
            )
        sleep(config.poll_interval_seconds)


@overload
def resolve_secret(env: Env, name: str, key: str, default: str) -> str: ...


@overload
def resolve_secret(env: Env, name: str, key: str, default: None) -> Optional[str]: ...


def resolve_secret(env: Env, name: str, key: str, default: Optional[str]) -> Optional[str]:
    """Return the `key` secret from the mount, or the `name` environment variable when there is none.

    `env` must be the prefixed `Env` of the calling config, so that the fallback reads exactly the same
    variable as before.
    """
    value = get_secrets().get(key)
    return value if value is not None else env.str(name=name, default=default)


def resolve_secret_list(env: Env, name: str, key: str, default: list[str]) -> list[str]:
    """Comma-separated variant of `resolve_secret`, matching `Env.list`."""
    value = get_secrets().get(key)
    if value is None:
        return env.list(name=name, default=default)
    return [item.strip() for item in value.split(",") if item.strip()]
