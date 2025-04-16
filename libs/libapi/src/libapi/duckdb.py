# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import errno
import json
import logging
import os
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional

import anyio
import duckdb
from datasets import Features
from filelock import AsyncFileLock
from huggingface_hub import HfApi
from libcommon.constants import CONFIG_PARQUET_METADATA_KIND, PARQUET_REVISION, SPLIT_DUCKDB_INDEX_KIND
from libcommon.duckdb_utils import (
    CREATE_INDEX_COMMAND,
    CREATE_INDEX_ID_COLUMN_COMMANDS,
    CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES,
    CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND_FROM_LIST_OF_PARQUET_FILES,
    DUCKDB_DEFAULT_INDEX_FILENAME,
    DUCKDB_DEFAULT_PARTIAL_INDEX_FILENAME,
    INSTALL_AND_LOAD_EXTENSION_COMMAND,
    SET_EXTENSIONS_DIRECTORY_COMMAND,
    compute_transformed_data,
    get_indexable_columns,
    get_monolingual_stemmer,
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
        split_directory = extract_split_directory_from_parquet_url(url)
        repo_file_location = f"{config}/{split_directory}/{filename}"
        index_file_location = f"{index_folder}/{repo_file_location}"
        index_path = anyio.Path(index_file_location)
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


def build_index_file(
    cache_folder: StrPath,
    index_folder: StrPath,
    dataset: str,
    config: str,
    split: str,
    repo_file_location: str,
    hf_token: Optional[str],
    max_split_size_bytes: int,
    extensions_directory: Optional[str],
    parquet_metadata_directory: StrPath,
    split_parquet_files: list[ParquetFileMetadataItem],
    features: dict[str, Any],
) -> None:
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

    column_names = ",".join(f'"{column}"' for column in features)

    # look for indexable columns (= possibly nested columns containing string data)
    indexable_columns = ",".join(f'"{column}"' for column in get_indexable_columns(Features.from_dict(features)))

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
                    resume_download=False,
                )
            )
        )

    transformed_df = None
    try:
        transformed_df = compute_transformed_data(all_split_parquets, features)
    except Exception as err:
        logging.info(f"Unable to compute transformed data {err}, skipping statistics.")

    # create index
    index_file_location = f"{index_folder}/{repo_file_location}"
    Path(index_file_location).parent.mkdir(exist_ok=True, parents=True)
    con = duckdb.connect(str(Path(index_file_location).resolve()))

    hf_api = HfApi(token=hf_token)
    stemmer = None

    try:
        if transformed_df is not None:
            logging.debug(transformed_df.head())
            # update original data with results of transformations (string lengths, audio durations, etc.):
            logging.info(f"Updating data with {transformed_df.columns}")
            create_command_sql = CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND_FROM_LIST_OF_PARQUET_FILES.format(
                columns=column_names, source=[str(p) for p in all_split_parquets]
            )

        else:
            create_command_sql = CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES.format(
                columns=column_names, source=[str(p) for p in all_split_parquets]
            )

        logging.info(create_command_sql)
        con.sql(create_command_sql)
        con.sql(CREATE_INDEX_ID_COLUMN_COMMANDS)
        logging.debug(con.sql("SELECT * FROM data LIMIT 5;"))
        logging.debug(con.sql("SELECT count(*) FROM data;"))

        if len(indexable_columns) > 0:
            # configure duckdb extensions
            if extensions_directory is not None:
                con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))
            con.execute(INSTALL_AND_LOAD_EXTENSION_COMMAND)
            stemmer = get_monolingual_stemmer(hf_api.dataset_info(repo_id=dataset).card_data)
            create_index_sql = CREATE_INDEX_COMMAND.format(columns=indexable_columns, stemmer=stemmer)
            logging.info(create_index_sql)
            con.sql(create_index_sql)

        # make sure there is no WAL at the end
        con.sql("CHECKPOINT;")

    finally:
        con.close()


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

        # use a lock in case another worker is currently writing
        cache_folder = Path(duckdb_index_file_directory) / HUB_DOWNLOAD_CACHE_FOLDER
        cache_folder.mkdir(exist_ok=True, parents=True)
        async with AsyncFileLock(cache_folder / ".lock", timeout=60):
            if not await index_path.is_file():
                with StepProfiler(method="get_index_file_location_and_build_if_missing", step="build duckdb index"):
                    await anyio.to_thread.run_sync(
                        build_index_file,
                        cache_folder,
                        index_folder,
                        dataset,
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
        # Update its modification time
        await index_path.touch()
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
