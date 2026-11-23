# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from huggingface_hub import BucketFile, HfApi

# The search pods copy a file under a temporary name, then rename it: one left behind is a copy that was interrupted.
TEMPORARY_FILE_SUFFIX = ".tmp"
TEMPORARY_FILE_EXPIRED_SECONDS = 3600
# Reading an index changes nothing in the bucket: the search pods rewrite this file next to each index they serve (see
# SHARED_INDEX_USED_SUFFIX in libapi.duckdb).
USED_SUFFIX = ".used"
DELETE_BATCH_SIZE = 1000


def clean_shared_index_cache(
    hf_endpoint: str, bucket: str, hf_token: Optional[str], expired_time_interval_seconds: int
) -> None:
    """Delete from the shared index cache of search the indexes no pod served for `expired_time_interval_seconds`.

    The last use of an index is the upload time of its `.used` file, or of the index itself when it has none. The next
    search pod that misses a deleted index builds it again and uploads it again.
    """
    api = HfApi(endpoint=hf_endpoint, token=hf_token)
    now = datetime.now(timezone.utc)
    files = {
        entry.path: entry for entry in api.list_bucket_tree(bucket, recursive=True) if isinstance(entry, BucketFile)
    }

    def last_use(path: str) -> datetime:
        times = [
            entry.uploaded_at or entry.mtime or now
            for entry in (files.get(path), files.get(path + USED_SUFFIX))
            if entry
        ]
        return max(times, default=now)

    expired: list[str] = []
    for path in files:
        if path.endswith(TEMPORARY_FILE_SUFFIX):
            if now - last_use(path) > timedelta(seconds=TEMPORARY_FILE_EXPIRED_SECONDS):
                expired.append(path)
        elif path.endswith(USED_SUFFIX):
            # deleted with its index, or alone when its index is gone
            index = path.removesuffix(USED_SUFFIX)
            if index not in files and now - last_use(index) > timedelta(seconds=expired_time_interval_seconds):
                expired.append(path)
        elif now - last_use(path) > timedelta(seconds=expired_time_interval_seconds):
            expired.append(path)
            if path + USED_SUFFIX in files:
                expired.append(path + USED_SUFFIX)

    for start in range(0, len(expired), DELETE_BATCH_SIZE):
        api.batch_bucket_files(bucket, delete=expired[start : start + DELETE_BATCH_SIZE])
    deleted = set(expired)
    deleted_bytes = sum(files[path].size for path in deleted)
    kept_bytes = sum(entry.size for path, entry in files.items() if path not in deleted)
    logging.info(
        f"deleted {len(deleted)} files ({deleted_bytes >> 30} GiB) from {bucket}, kept {len(files) - len(deleted)} files"
        f" ({kept_bytes >> 30} GiB)"
    )
