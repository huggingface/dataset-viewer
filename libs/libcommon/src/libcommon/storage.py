# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import os
import shutil
from datetime import datetime, timedelta
from os import PathLike, makedirs
from pathlib import Path
from typing import Optional, Union

from appdirs import user_cache_dir  # type:ignore

from libcommon.constants import (
    DESCRIPTIVE_STATISTICS_CACHE_APPNAME,
    DUCKDB_INDEX_CACHE_APPNAME,
    HF_DATASETS_CACHE_APPNAME,
    PARQUET_METADATA_CACHE_APPNAME,
)

StrPath = Union[str, PathLike[str]]


def init_dir(directory: Optional[StrPath] = None, appname: Optional[str] = None) -> StrPath:
    """Initialize a directory.

    If directory is None, it will be set to the default cache location on the machine (using appname as a key, if
    not None).

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.
        appname (`str`, *optional*): The name of the application. Used if `directory`is None. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    if directory is None:
        directory = user_cache_dir(appname=appname)
        logging.debug(f"Directory defaulting to user-specific cache: {directory}")
    makedirs(directory, exist_ok=True)
    logging.debug(f"Directory created at: {directory}")
    return directory


def init_parquet_metadata_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the parquet metadata directory.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=PARQUET_METADATA_CACHE_APPNAME)


def init_duckdb_index_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the duckdb index directory.

    If directory is None, it will be set to the default duckdb index location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=DUCKDB_INDEX_CACHE_APPNAME)


def init_hf_datasets_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the cache directory for the datasets library.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=HF_DATASETS_CACHE_APPNAME)


def init_statistics_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the cache directory for storage of a dataset in parquet format for statistics computations.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=DESCRIPTIVE_STATISTICS_CACHE_APPNAME)


def exists(path: StrPath) -> bool:
    """Check if a path exists.

    Args:
        path (`StrPath`): The path to check.

    Returns:
        `bool`: True if the path exists, False otherwise.
    """
    return Path(path).exists()


def remove_dir(directory: StrPath) -> None:
    """Remove a directory.

    If the directory does not exist, don't raise.

    Args:
        directory (`StrPath`): The directory to remove.
    """
    shutil.rmtree(directory, ignore_errors=True)
    logging.debug(f"Directory removed: {directory}")


def clean_dir(root_folder: StrPath, expired_time_interval_seconds: int) -> None:
    """
    Delete temporary cache directories under a root folder.
    """
    logging.info(
        f"looking for all files and directories under {root_folder} to delete files with last accessed time before {expired_time_interval_seconds} seconds ago or is empty folder"
    )
    now = datetime.now().replace(tzinfo=None)
    errors = 0
    total_dirs = 0
    total_files = 0
    total_disappeared_paths = 0

    for root, _, files in os.walk(root_folder, topdown=False):
        try:
            for name in files:
                path = os.path.join(root, name)
                last_access_time_value = os.path.getatime(path)
                last_access_datetime = datetime.fromtimestamp(last_access_time_value).replace(tzinfo=None)
                if last_access_datetime + timedelta(seconds=expired_time_interval_seconds) <= now:
                    logging.info(f"deleting file {path=} {last_access_datetime=}")
                    os.remove(path)
                    total_files += 1
            try:
                os.rmdir(root)
                logging.info(f"deleting directory {root=} because it was empty")
                total_dirs += 1
            except OSError:
                pass  # Ignore non-empty directories
        except FileNotFoundError:
            logging.error(f"failed to delete {path=} because it has disappeared during the loop")
            total_disappeared_paths += 1
    if total_files:
        logging.info(f"clean_directory removed {total_files} files at the root of the cache directory.")
    if total_disappeared_paths:
        logging.info(
            f"clean_directory failed to delete {total_disappeared_paths} paths because they disappeared during the loop."
        )

    logging.info(f"clean_directory removed {total_dirs - errors} directories at the root of the cache directory.")
