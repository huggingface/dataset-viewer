# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from huggingface_hub import BucketFile, BucketFolder

from cache_maintenance import shared_index_cache
from cache_maintenance.shared_index_cache import clean_shared_index_cache

INDEX = "v1-duckdb1.5.5/downloads/{}/default/train/index.duckdb"


def bucket_file(path: str, age: timedelta) -> BucketFile:
    uploaded_at = (datetime.now(timezone.utc) - age).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # the format of the Hub API
    return BucketFile(type="file", path=path, size=2**30, xetHash="x", mtime=uploaded_at, uploadedAt=uploaded_at)  # type: ignore


def test_clean_shared_index_cache() -> None:
    entries = [
        BucketFolder(type="directory", path="v1-duckdb1.5.5"),  # type: ignore
        # not served for 3 days: deleted with its marker
        bucket_file(INDEX.format("unused"), timedelta(days=5)),
        bucket_file(INDEX.format("unused") + ".used", timedelta(days=3)),
        # uploaded 5 days ago, served an hour ago: kept
        bucket_file(INDEX.format("used"), timedelta(days=5)),
        bucket_file(INDEX.format("used") + ".used", timedelta(hours=1)),
        # never marked: its upload time counts
        bucket_file(INDEX.format("old"), timedelta(days=3)),
        bucket_file(INDEX.format("recent"), timedelta(hours=1)),
        # a marker whose index is gone
        bucket_file(INDEX.format("gone") + ".used", timedelta(days=3)),
        # an interrupted copy, and a copy in progress
        bucket_file(INDEX.format("recent") + ".pod.1.tmp", timedelta(hours=2)),
        bucket_file(INDEX.format("recent") + ".pod.2.tmp", timedelta(minutes=5)),
    ]
    deleted: list[list[str]] = []

    def batch_bucket_files(_: Any, bucket_id: str, *, delete: list[str]) -> None:
        assert bucket_id == "org/bucket"
        deleted.append(delete)

    with (
        patch("huggingface_hub.HfApi.list_bucket_tree", lambda *_, **__: iter(entries)),
        patch("huggingface_hub.HfApi.batch_bucket_files", batch_bucket_files),
        patch.object(shared_index_cache, "DELETE_BATCH_SIZE", 2),
    ):
        clean_shared_index_cache(
            hf_endpoint="https://huggingface.co",
            bucket="org/bucket",
            hf_token=None,
            expired_time_interval_seconds=172_800,
        )

    assert [len(batch) for batch in deleted] == [2, 2, 1]
    assert {path for batch in deleted for path in batch} == {
        INDEX.format("unused"),
        INDEX.format("unused") + ".used",
        INDEX.format("old"),
        INDEX.format("gone") + ".used",
        INDEX.format("recent") + ".pod.1.tmp",
    }
