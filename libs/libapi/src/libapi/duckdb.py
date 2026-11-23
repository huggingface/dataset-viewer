# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import time
import traceback
from collections.abc import Coroutine
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional, TypeVar

import anyio
import duckdb
from datasets import Features
from duckdb import __version__ as duckdb_version
from filelock import AsyncFileLock
from huggingface_hub import HfApi
from libcommon.constants import CONFIG_PARQUET_METADATA_KIND, PARQUET_REVISION, ROW_IDX_COLUMN
from libcommon.duckdb_utils import (
    CREATE_INDEX_ID_COLUMN_COMMANDS,
    CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES,
    CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND_FROM_LIST_OF_PARQUET_FILES,
    DUCKDB_BUILD_CONFIG,
    DUCKDB_DEFAULT_INDEX_FILENAME,
    DUCKDB_DEFAULT_PARTIAL_INDEX_FILENAME,
    compute_transformed_data,
    create_index,
    duckdb_connect,
    get_indexable_columns,
    get_monolingual_stemmer,
    key_sql,
    varchar_sql,
)
from libcommon.parquet_utils import (
    ParquetFileMetadataItem,
    extract_split_directory_from_parquet_url,
    get_num_parquet_files_to_process,
    parquet_export_is_partial,
)
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CacheEntry
from libcommon.storage import StrPath, init_dir
from libcommon.storage_client import StorageClient
from libcommon.utils import download_file_from_hub

from libapi.exceptions import DownloadIndexError, ResponseNotReadyError
from libapi.utils import get_cache_entry_from_step

REPO_TYPE = "dataset"
DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY = "downloads"
HUB_DOWNLOAD_CACHE_FOLDER = "cache"

T = TypeVar("T")

_all_tasks: list[asyncio.Task[Any]] = []  # kinda like threading.all_threads()
_index_builds: dict[str, asyncio.Task[None]] = {}
INDEX_BUILD_WAIT_SECONDS = 20
# builds get their own threads, so they never take the default ones the queries and the Hub auth check need
MAX_CONCURRENT_INDEX_BUILDS = 6
_index_build_limiter = anyio.CapacityLimiter(MAX_CONCURRENT_INDEX_BUILDS)
# copies from and to the shared index cache are I/O: they don't take a build slot
_shared_cache_limiter = anyio.CapacityLimiter(4)
# bump when the build changes the content of an index: indexes shared by older builds are then ignored
SHARED_INDEX_FORMAT_VERSION = 1
# Reading an index changes nothing in the bucket, and the pods read it from their local copy: they rewrite this file
# next to the shared index instead, so that the cleaning job keeps the indexes that are still used.
SHARED_INDEX_USED_SUFFIX = ".used"
SHARED_INDEX_USED_INTERVAL_SECONDS = 6 * 3600
_shared_index_marked_at: dict[str, float] = {}
# a queued build that nobody requested for this long is dropped when its turn comes: the next request starts it again.
# The last request time is the mtime of a file next to the index, so that every uvicorn worker of the pod sees it.
INDEX_BUILD_STALE_SECONDS = 120
# a build that failed in DuckDB (out of memory...) fails again the same way: its requests get the same error for this
# long instead of starting another build. Other errors (Hub downloads...) can be transient and are retried right away.
FAILED_INDEX_BUILD_RETRY_SECONDS = 600
_failed_index_builds: dict[str, tuple[float, duckdb.Error]] = {}


def _handle_errors(task: asyncio.Task[Any]) -> None:
    try:
        exc = task.exception()  # Also marks that the exception has been handled
        if exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
    except asyncio.exceptions.CancelledError:
        pass


def _task_done(task: asyncio.Task[Any]) -> None:
    _all_tasks.remove(task)
    _handle_errors(task)


def create_task(awaitable: Coroutine[None, None, T]) -> asyncio.Task[T]:
    """
    Spawn an awaitable as a stand-alone task.
    This allows for fire-and-forget with errors logging.
    """
    task = asyncio.create_task(awaitable)
    _all_tasks.append(task)
    task.add_done_callback(_task_done)
    return task


def build_index_file(
    cache_folder: StrPath,
    index_folder: StrPath,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    repo_file_location: str,
    hf_token: Optional[str],
    max_split_size_bytes: int,
    extensions_directory: Optional[str],
    parquet_metadata_directory: StrPath,
    split_parquet_files: list[ParquetFileMetadataItem],
    features: dict[str, Any],
) -> bool:
    """Returns False when the index was built without its transformed columns (string lengths, audio durations...)."""
    logging.info(f"compute and cache duckdb index on-the-fly for {dataset=} {config=} {split=}")
    if not split_parquet_files:
        raise DownloadIndexError("No parquet files found.")

    # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
    # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
    split_directory = extract_split_directory_from_parquet_url(split_parquet_files[0]["url"])

    num_parquet_files_to_index, num_bytes, num_rows = get_num_parquet_files_to_process(
        parquet_files=split_parquet_files,
        parquet_metadata_directory=parquet_metadata_directory,
        max_size_bytes=max_split_size_bytes,
    )

    split_parquet_files = split_parquet_files[:num_parquet_files_to_index]
    parquet_file_names = [parquet_file["filename"] for parquet_file in split_parquet_files]

    column_names_sql = ",".join(key_sql(column) for column in features)

    # look for indexable columns (= possibly nested columns containing string data)
    indexable_columns = get_indexable_columns(Features.from_dict(features))

    all_split_parquets: list[Path] = []
    for parquet_file in parquet_file_names:
        all_split_parquets.append(
            Path(
                download_file_from_hub(
                    repo_type="dataset",
                    revision=PARQUET_REVISION,
                    repo_id=dataset,
                    filename=f"{config}/{split_directory}/{parquet_file}",
                    hf_token=hf_token,
                    cache_dir=cache_folder,
                )
            )
        )

    transformed_df = None
    complete = True
    try:
        transformed_df = compute_transformed_data(
            all_split_parquets, features, hf_token=hf_token, repo_id=dataset, hash=revision
        )
    except Exception as err:
        logging.info(f"Unable to compute transformed data {err}, skipping statistics.")
        complete = False

    # create index
    index_file_location = f"{index_folder}/{repo_file_location}"
    Path(index_file_location).parent.mkdir(exist_ok=True, parents=True)
    # Readers only look for the final path: build in a temporary file and rename it once complete, so that no
    # request opens (and then deletes as corrupted) an index that is still being written.
    tmp_index_file_location = f"{index_file_location}.tmp"

    try:
        with duckdb_connect(
            database=tmp_index_file_location,
            extensions_directory=extensions_directory,
            allowed_paths=all_split_parquets,
            transformed_df=transformed_df,
            config={**DUCKDB_BUILD_CONFIG, "temp_directory": f"{tmp_index_file_location}.spill"},
        ) as con:
            if transformed_df is not None:
                logging.debug(transformed_df.head())
                # update original data with results of transformations (string lengths, audio durations, etc.):
                logging.info(f"Updating data with {transformed_df.columns}")
                create_command_sql = CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND_FROM_LIST_OF_PARQUET_FILES.format(
                    columns=column_names_sql,
                    source="[" + ",".join(varchar_sql(str(p)) for p in all_split_parquets) + "]",
                )

            else:
                create_command_sql = CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES.format(
                    columns=column_names_sql,
                    source="[" + ",".join(varchar_sql(str(p)) for p in all_split_parquets) + "]",
                )

            logging.info(create_command_sql)
            con.sql(create_command_sql)
            con.sql(CREATE_INDEX_ID_COLUMN_COMMANDS)
            logging.debug(con.sql("SELECT * FROM data LIMIT 5;"))
            logging.debug(con.sql("SELECT count(*) FROM data;"))
            # make sure there is no WAL at the end
            con.sql("CHECKPOINT;")

        if len(indexable_columns) > 0:
            stemmer = get_monolingual_stemmer(HfApi(token=hf_token).dataset_info(repo_id=dataset).card_data)
            logging.info(f"Indexing {dataset} ({config=}, {split=})")
            create_index(
                database=tmp_index_file_location,
                input_table="data",
                columns=indexable_columns,
                stemmer=stemmer,
                input_id=ROW_IDX_COLUMN,
                fts_schema="fts_main_data",
                extensions_directory=extensions_directory,
            )
            logging.info(f"Done indexing {dataset} ({config=}, {split=})")
        os.replace(tmp_index_file_location, index_file_location)
    except Exception:
        Path(tmp_index_file_location).unlink(missing_ok=True)
        raise
    return complete


def _copy_file_atomically(source: str, destination: str) -> None:
    # readers only look for the final name: a pod that reads the shared cache never sees a partial copy
    Path(destination).parent.mkdir(exist_ok=True, parents=True)
    tmp_destination = f"{destination}.{socket.gethostname()}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, tmp_destination)
        os.replace(tmp_destination, destination)
    finally:
        Path(tmp_destination).unlink(missing_ok=True)


def _copy_from_shared_cache(shared_index_file_location: str, index_file_location: str) -> bool:
    try:
        _copy_file_atomically(shared_index_file_location, index_file_location)
    except FileNotFoundError:
        return False
    except OSError as err:
        logging.warning(f"failed to copy {shared_index_file_location} from the shared index cache: {err}")
        return False
    logging.info(f"copied {index_file_location} from the shared index cache")
    return True


def _copy_to_shared_cache(index_file_location: str, shared_index_file_location: str) -> None:
    try:
        _copy_file_atomically(index_file_location, shared_index_file_location)
    except OSError as err:
        logging.warning(f"failed to copy {index_file_location} to the shared index cache: {err}")


def _mark_shared_index_used(shared_index_file_location: str) -> None:
    try:
        # an index that is not shared (incomplete, or still being uploaded) has nothing to keep
        if os.path.isfile(shared_index_file_location):
            Path(shared_index_file_location + SHARED_INDEX_USED_SUFFIX).write_text(str(time.time()))
    except OSError as err:
        logging.warning(f"failed to mark {shared_index_file_location} as used in the shared index cache: {err}")


def _build_index_file_unless_stale(index_file_location: str, *build_args: Any) -> bool:
    """Returns True when it built a complete index."""
    try:
        last_requested = os.path.getmtime(index_file_location + ".requested")
    except FileNotFoundError:
        last_requested = 0
    if time.time() - last_requested > INDEX_BUILD_STALE_SECONDS:
        logging.info(
            f"skip the build of {index_file_location}: not requested in the last {INDEX_BUILD_STALE_SECONDS}s"
        )
        return False
    return build_index_file(*build_args)


async def _build_index_file_once(
    index_file_location: str, shared_index_file_location: Optional[str], *build_args: Any
) -> None:
    # The lock is held for the whole build, so that the other uvicorn workers of the pod wait for it (in their event
    # loop, without holding a thread) instead of building again.
    built = False
    async with AsyncFileLock(index_file_location + ".lock"):
        if await anyio.Path(index_file_location).is_file():
            return
        if shared_index_file_location:
            copied = await anyio.to_thread.run_sync(
                _copy_from_shared_cache, shared_index_file_location, index_file_location, limiter=_shared_cache_limiter
            )
            if copied:
                return
        built = await anyio.to_thread.run_sync(
            _build_index_file_unless_stale, index_file_location, *build_args, limiter=_index_build_limiter
        )
    if built and shared_index_file_location:
        # after the lock: the requests are served from the local index while it is copied for the other pods
        create_task(
            anyio.to_thread.run_sync(
                _copy_to_shared_cache, index_file_location, shared_index_file_location, limiter=_shared_cache_limiter
            )
        )


def _end_index_build(index_file_location: str, build: asyncio.Task[None]) -> None:
    _index_builds.pop(index_file_location, None)
    error = None if build.cancelled() else build.exception()
    if isinstance(error, duckdb.Error):
        _failed_index_builds[index_file_location] = (time.time(), error)


async def get_index_file_location_and_build_if_missing(
    duckdb_index_file_directory: StrPath,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    hf_token: Optional[str],
    max_split_size_bytes: int,
    extensions_directory: Optional[str],
    parquet_metadata_directory: StrPath,
    split_parquet_files: list[ParquetFileMetadataItem],
    features: dict[str, Any],
    shared_index_directory: Optional[StrPath] = None,
) -> tuple[str, bool]:
    with StepProfiler(method="get_index_file_location_and_build_if_missing", step="all"):
        # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
        # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
        split_directory = extract_split_directory_from_parquet_url(split_parquet_files[0]["url"])
        partial_parquet_export = parquet_export_is_partial(split_parquet_files[0]["url"])

        num_parquet_files_to_index, num_bytes, num_rows = get_num_parquet_files_to_process(
            parquet_files=split_parquet_files,
            parquet_metadata_directory=parquet_metadata_directory,
            max_size_bytes=max_split_size_bytes,
        )

        index_filename = (
            DUCKDB_DEFAULT_PARTIAL_INDEX_FILENAME
            if (num_parquet_files_to_index < len(split_parquet_files))
            else DUCKDB_DEFAULT_INDEX_FILENAME
        )
        partial = partial_parquet_export or (num_parquet_files_to_index < len(split_parquet_files))
        repo_file_location = f"{config}/{split_directory}/{index_filename}"

        # We can expect the duckdb index to be around the same size as the num_bytes.
        # But we apply a factor since we also add the full-text-search index and additional columns
        size_factor = 5
        index_folder = get_download_folder(
            duckdb_index_file_directory, size_factor * num_bytes, dataset, config, split, revision
        )
        index_file_location = f"{index_folder}/{repo_file_location}"
        index_path = anyio.Path(index_file_location)

        shared_index_file_location = None
        if shared_index_directory:
            shared_index_file_location = os.path.join(
                shared_index_directory,
                f"v{SHARED_INDEX_FORMAT_VERSION}-duckdb{duckdb_version}",
                os.path.relpath(index_file_location, duckdb_index_file_directory),
            )
        Path(index_file_location).parent.mkdir(exist_ok=True, parents=True)
        if not await index_path.is_file():
            await anyio.Path(index_file_location + ".requested").touch()
            # One build per index: the requests that come while it runs wait for the same task instead of starting
            # another build. The task is shielded, so it goes on after a request gives up waiting.
            build = _index_builds.get(index_file_location)
            failure = _failed_index_builds.get(index_file_location)
            if build is None and failure:
                failed_at, error = failure
                if time.time() - failed_at < FAILED_INDEX_BUILD_RETRY_SECONDS:
                    raise error.with_traceback(None)
                del _failed_index_builds[index_file_location]
            if build is None:
                cache_folder = Path(duckdb_index_file_directory) / HUB_DOWNLOAD_CACHE_FOLDER
                cache_folder.mkdir(exist_ok=True, parents=True)
                build = create_task(
                    _build_index_file_once(
                        index_file_location,
                        shared_index_file_location,
                        cache_folder,
                        index_folder,
                        dataset,
                        revision,
                        config,
                        split,
                        repo_file_location,
                        hf_token,
                        max_split_size_bytes,
                        extensions_directory,
                        parquet_metadata_directory,
                        split_parquet_files,
                        features,
                    )
                )
                _index_builds[index_file_location] = build
                build.add_done_callback(lambda task: _end_index_build(index_file_location, task))
            with StepProfiler(method="get_index_file_location_and_build_if_missing", step="build duckdb index"):
                try:
                    await asyncio.wait_for(asyncio.shield(build), timeout=INDEX_BUILD_WAIT_SECONDS)
                except asyncio.TimeoutError:
                    _msg = "this can take a minute" if len(_all_tasks) < 20 else "this may take longer than usual"
                    raise ResponseNotReadyError("the dataset index is loading, " + _msg)
        # Update its modification time
        await index_path.touch()
        if shared_index_file_location:
            marked_at = _shared_index_marked_at.get(shared_index_file_location, 0)
            if time.time() - marked_at > SHARED_INDEX_USED_INTERVAL_SECONDS:
                _shared_index_marked_at[shared_index_file_location] = time.time()
                create_task(
                    anyio.to_thread.run_sync(
                        _mark_shared_index_used, shared_index_file_location, limiter=_shared_cache_limiter
                    )
                )
        return index_file_location, partial


def get_download_folder(
    root_directory: StrPath, size_bytes: int, dataset: str, revision: str, config: str, split: str
) -> str:
    check_available_disk_space(root_directory, size_bytes)
    payload = (dataset, config, split, revision)
    hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
    subdirectory = "".join([c if re.match(r"[\w-]", c) else "-" for c in f"{dataset}-{hash_suffix}"])
    return f"{root_directory}/{DUCKDB_INDEX_DOWNLOADS_SUBDIRECTORY}/{subdirectory}"


def check_available_disk_space(path: StrPath, required_space: int) -> None:
    try:
        disk_stat = os.statvfs(path)
    except FileNotFoundError:
        # The path does not exist, we create it and
        init_dir(path)
        disk_stat = os.statvfs(path)
    # Calculate free space in bytes
    free_space = disk_stat.f_bavail * disk_stat.f_frsize
    logging.debug(f"{free_space} available space, needed {required_space}")
    if free_space < required_space:
        raise DownloadIndexError(
            "Cannot perform the search due to a lack of disk space on the server. Please report the issue."
        )


def get_cache_entry_from_parquet_metadata_job(
    dataset: str,
    config: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    hf_timeout_seconds: Optional[float],
    blocked_datasets: list[str],
    storage_clients: Optional[list[StorageClient]] = None,
) -> CacheEntry:
    return get_cache_entry_from_step(
        processing_step_name=CONFIG_PARQUET_METADATA_KIND,
        dataset=dataset,
        config=config,
        split=None,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        blocked_datasets=blocked_datasets,
        storage_clients=storage_clients,
    )
