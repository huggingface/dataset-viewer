# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import errno
import json
import logging
import os
import re
from hashlib import sha1
from typing import Optional

import anyio
from anyio import Path
from libcommon.constants import SPLIT_DUCKDB_INDEX_KIND
from libcommon.parquet_utils import extract_split_name_from_parquet_url
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CacheEntry
from libcommon.storage import StrPath, init_dir
from libcommon.storage_client import StorageClient
from libcommon.utils import download_file_from_hub

from libapi.exceptions import DownloadIndexError
from libapi.utils import get_cache_entry_from_step

REPO_TYPE = "dataset"
DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY = "downloads"
HUB_DOWNLOAD_CACHE_FOLDER = "cache"


async def get_index_file_location_and_download_if_missing(
    duckdb_index_file_directory: StrPath,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    filename: str,
    size_bytes: int,
    url: str,
    target_revision: str,
    hf_token: Optional[str],
) -> str:
    with StepProfiler(method="get_index_file_location_and_download_if_missing", step="all"):
        index_folder = get_download_folder(duckdb_index_file_directory, size_bytes, dataset, config, split, revision)
        # For directories like "partial-train" for the file
        # at "en/partial-train/0000.parquet" in the C4 dataset.
        # Note that "-" is forbidden for split names, so it doesn't create directory names collisions.
        split_directory = extract_split_name_from_parquet_url(url)
        repo_file_location = f"{config}/{split_directory}/{filename}"
        index_file_location = f"{index_folder}/{repo_file_location}"
        index_path = Path(index_file_location)
        if not await index_path.is_file():
            with StepProfiler(method="get_index_file_location_and_download_if_missing", step="download index file"):
                cache_folder = f"{duckdb_index_file_directory}/{HUB_DOWNLOAD_CACHE_FOLDER}"
                await anyio.to_thread.run_sync(
                    download_index_file,
                    cache_folder,
                    index_folder,
                    target_revision,
                    dataset,
                    repo_file_location,
                    hf_token,
                )
        # Update its modification time
        await index_path.touch()
        return index_file_location


def get_download_folder(
    root_directory: StrPath, size_bytes: int, dataset: str, revision: str, config: str, split: str
) -> str:
    check_available_disk_space(root_directory, size_bytes)
    payload = (dataset, config, split, revision)
    hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
    subdirectory = "".join([c if re.match(r"[\w-]", c) else "-" for c in f"{dataset}-{hash_suffix}"])
    return f"{root_directory}/{DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY}/{subdirectory}"


def check_available_disk_space(path: StrPath, required_space: int) -> None:
    disk_stat = os.statvfs(path)
    # Calculate free space in bytes
    free_space = disk_stat.f_bavail * disk_stat.f_frsize
    logging.debug(f"{free_space} available space, needed {required_space}")
    if free_space < required_space:
        raise DownloadIndexError(
            "Cannot perform the search due to a lack of disk space on the server. Please report the issue."
        )


def download_index_file(
    cache_folder: str,
    index_folder: str,
    target_revision: str,
    dataset: str,
    repo_file_location: str,
    hf_token: Optional[str] = None,
) -> None:
    logging.info(f"init_dir {index_folder}")
    try:
        init_dir(index_folder)
        download_file_from_hub(
            repo_type=REPO_TYPE,
            revision=target_revision,
            repo_id=dataset,
            filename=repo_file_location,
            local_dir=index_folder,
            hf_token=hf_token,
            cache_dir=cache_folder,
        )
    except OSError as err:
        if err.errno == errno.ENOSPC:
            raise DownloadIndexError(
                "Cannot perform the operation due to a lack of disk space on the server. Please report the issue.", err
            )


def get_cache_entry_from_duckdb_index_job(
    dataset: str,
    config: str,
    split: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    hf_timeout_seconds: Optional[float],
    blocked_datasets: list[str],
    storage_clients: Optional[list[StorageClient]] = None,
) -> CacheEntry:
    return get_cache_entry_from_step(
        processing_step_name=SPLIT_DUCKDB_INDEX_KIND,
        dataset=dataset,
        config=config,
        split=split,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        blocked_datasets=blocked_datasets,
        storage_clients=storage_clients,
    )
