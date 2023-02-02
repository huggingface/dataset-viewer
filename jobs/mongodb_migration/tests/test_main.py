# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest

from mongodb_migration.main import run_job


@pytest.mark.parametrize(
    "cache_url,queue_url,migration_url",
    [
        (None, None, None),
        ("mongodb://doesnotexist:123", None, None),
        (None, "mongodb://doesnotexist:123", None),
        (None, None, "mongodb://doesnotexist:123"),
    ],
)
def test_run_job(cache_url: Optional[str], queue_url: Optional[str], migration_url: Optional[str]) -> None:
    monkeypatch = pytest.MonkeyPatch()
    timeout_ms = "3_000"
    monkeypatch.setenv("CACHE_MONGO_CONNECTION_TIMEOUT_MS", timeout_ms)
    monkeypatch.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    if cache_url:
        monkeypatch.setenv("CACHE_MONGO_URL", cache_url)
    monkeypatch.setenv("QUEUE_MONGO_CONNECTION_TIMEOUT_MS", timeout_ms)
    monkeypatch.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    if queue_url:
        monkeypatch.setenv("QUEUE_MONGO_URL", queue_url)
    monkeypatch.setenv("MONGODB_MIGRATION_MONGO_CONNECTION_TIMEOUT_MS", timeout_ms)
    monkeypatch.setenv("MONGODB_MIGRATION_MONGO_DATABASE", "datasets_server_maintenance_test")
    if migration_url:
        monkeypatch.setenv("MONGODB_MIGRATION_MONGO_URL", migration_url)
    run_job()
