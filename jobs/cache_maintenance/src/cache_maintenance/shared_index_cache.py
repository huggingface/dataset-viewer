# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from huggingface_hub import BucketFile, HfApi

# the search pods copy a file under a temporary name, then rename it: one left behind is a copy that was interrupted
TEMPORARY_FILE_SUFFIX = ".tmp"
TEMPORARY_FILE_EXPIRED_SECONDS = 3600
DELETE_BATCH_SIZE = 1000


def clean_shared_index_cache(
    hf_endpoint: str, bucket: str, hf_token: Optional[str], expired_time_interval_seconds: int
) -> None:
    """Delete the indexes uploaded to the shared index cache of search more than `expired_time_interval_seconds` ago.

    Reading an index does not change it in the bucket, so the age is the upload time: the next search pod that misses
    a deleted index builds it again and uploads it again.
    """
    api = HfApi(endpoint=hf_endpoint, token=hf_token)
    now = datetime.now(timezone.utc)
    expired = []
    kept_files = kept_bytes = deleted_bytes = 0
    for entry in api.list_bucket_tree(bucket, recursive=True):
        if not isinstance(entry, BucketFile):
            continue
        uploaded_at = entry.uploaded_at or entry.mtime or now
        max_age = (
            TEMPORARY_FILE_EXPIRED_SECONDS
            if entry.path.endswith(TEMPORARY_FILE_SUFFIX)
            else expired_time_interval_seconds
        )
        if now - uploaded_at > timedelta(seconds=max_age):
            expired.append(entry.path)
            deleted_bytes += entry.size
        else:
            kept_files += 1
            kept_bytes += entry.size
    for start in range(0, len(expired), DELETE_BATCH_SIZE):
        api.batch_bucket_files(bucket, delete=expired[start : start + DELETE_BATCH_SIZE])
    logging.info(
        f"deleted {len(expired)} files ({deleted_bytes >> 30} GiB) from {bucket}, kept {kept_files} files"
        f" ({kept_bytes >> 30} GiB)"
    )
