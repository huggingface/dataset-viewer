# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import json
import logging
import os
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional

import anyio
import duckdb
from huggingface_hub import hf_hub_download
from libcommon.constants import DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CacheEntry
from libcommon.storage import StrPath, init_dir

from libapi.utils import get_cache_entry_from_steps

REPO_TYPE = "dataset"
HUB_DOWNLOAD_CACHE_FOLDER = "cache"


async def get_index_file_location_and_download_if_missing(
    duckdb_index_file_directory: StrPath,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    filename: str,
    url: str,
    target_revision: str,
    hf_token: Optional[str],
) -> str:
    with StepProfiler(method="get_index_file_location_and_download_if_missing", step="all"):
        index_folder = get_download_folder(duckdb_index_file_directory, dataset, config, split, revision)
        # For directories like "partial-train" for the file
        # at "en/partial-train/0000.parquet" in the C4 dataset.
        # Note that "-" is forbidden for split names, so it doesn't create directory names collisions.
        split_directory = url.rsplit("/", 2)[1]
        repo_file_location = f"{config}/{split_directory}/{filename}"
        index_file_location = f"{index_folder}/{repo_file_location}"
        index_path = Path(index_file_location)
        if not index_path.is_file():
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
        index_path.touch()
        return index_file_location


def get_download_folder(root_directory: StrPath, dataset: str, revision: str, config: str, split: str) -> str:
    payload = (dataset, config, split, revision)
    hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
    subdirectory = "".join([c if re.match(r"[\w-]", c) else "-" for c in f"{dataset}-{hash_suffix}"])
    return f"{root_directory}/{DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY}/{subdirectory}"


def download_index_file(
    cache_folder: str,
    index_folder: str,
    target_revision: str,
    dataset: str,
    repo_file_location: str,
    hf_token: Optional[str] = None,
) -> None:
    logging.info(f"init_dir {index_folder}")
    init_dir(index_folder)

    # see https://pypi.org/project/hf-transfer/ for more details about how to enable hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    hf_hub_download(
        repo_type=REPO_TYPE,
        revision=target_revision,
        repo_id=dataset,
        filename=repo_file_location,
        local_dir=index_folder,
        local_dir_use_symlinks=False,
        token=hf_token,
        cache_dir=cache_folder,
    )


def get_cache_entry_from_duckdb_index_job(
    processing_graph: ProcessingGraph,
    dataset: str,
    config: str,
    split: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    hf_timeout_seconds: Optional[float],
    cache_max_days: int,
    blocked_datasets: list[str],
) -> CacheEntry:
    processing_steps = processing_graph.get_processing_step_by_job_type("split-duckdb-index")
    return get_cache_entry_from_steps(
        processing_steps=[processing_steps],
        dataset=dataset,
        config=config,
        split=split,
        processing_graph=processing_graph,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        cache_max_days=cache_max_days,
        blocked_datasets=blocked_datasets,
    )


def duckdb_connect(**kwargs: Any) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(read_only=True, **kwargs)
    con.sql("SET enable_external_access=false;")
    con.sql("SET lock_configuration=true;")
    return con
