# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from huggingface_hub import BucketFile, BucketFolder

from cache_maintenance.shared_index_cache import DELETE_BATCH_SIZE, clean_shared_index_cache


def bucket_file(path: str, age: timedelta) -> BucketFile:
    uploaded_at = (datetime.now(timezone.utc) - age).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # the format of the Hub API
    return BucketFile(type="file", path=path, size=2**30, xetHash="x", mtime=uploaded_at, uploadedAt=uploaded_at)  # type: ignore


def test_clean_shared_index_cache() -> None:
    entries = [
        BucketFolder(type="directory", path="v1-duckdb1.5.5"),  # type: ignore
        bucket_file("v1-duckdb1.5.5/downloads/old/default/train/index.duckdb", timedelta(days=2)),
        bucket_file("v1-duckdb1.5.5/downloads/recent/default/train/index.duckdb", timedelta(hours=1)),
        # an interrupted copy
        bucket_file("v1-duckdb1.5.5/downloads/recent/default/test/index.duckdb.pod.1.tmp", timedelta(hours=2)),
        # a copy in progress
        bucket_file("v1-duckdb1.5.5/downloads/recent/default/dev/index.duckdb.pod.2.tmp", timedelta(minutes=5)),
    ] + [bucket_file(f"v1-duckdb1.5.5/downloads/many/{i}/index.duckdb", timedelta(days=3)) for i in range(1500)]
    deleted: list[list[str]] = []

    def batch_bucket_files(_: Any, bucket_id: str, *, delete: list[str]) -> None:
        assert bucket_id == "org/bucket"
        deleted.append(delete)

    with (
        patch("huggingface_hub.HfApi.list_bucket_tree", lambda *_, **__: iter(entries)),
        patch("huggingface_hub.HfApi.batch_bucket_files", batch_bucket_files),
    ):
        clean_shared_index_cache(
            hf_endpoint="https://huggingface.co",
            bucket="org/bucket",
            hf_token=None,
            expired_time_interval_seconds=86_400,
        )

    assert [len(batch) for batch in deleted] == [DELETE_BATCH_SIZE, 1502 - DELETE_BATCH_SIZE]
    paths = {path for batch in deleted for path in batch}
    assert "v1-duckdb1.5.5/downloads/old/default/train/index.duckdb" in paths
    assert "v1-duckdb1.5.5/downloads/recent/default/test/index.duckdb.pod.1.tmp" in paths
    assert "v1-duckdb1.5.5/downloads/recent/default/train/index.duckdb" not in paths
    assert "v1-duckdb1.5.5/downloads/recent/default/dev/index.duckdb.pod.2.tmp" not in paths
