# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import shutil
from os import PathLike, makedirs
from pathlib import Path
from typing import Optional, Union

from appdirs import user_cache_dir  # type:ignore

from libcommon.constants import (
    ASSETS_CACHE_APPNAME,
    CACHED_ASSETS_CACHE_APPNAME,
    DUCKDB_INDEX_CACHE_APPNAME,
    PARQUET_METADATA_CACHE_APPNAME,
)

StrPath = Union[str, PathLike[str]]


def init_dir(directory: Optional[StrPath] = None, appname: Optional[str] = None) -> StrPath:
    """Initialize a directory.

    If directory is None, it will be set to the default cache location on the machine (using appname as a key, if
    not None).

    Args:
        directory (Optional[Union[str, PathLike[str]]], optional): The directory to initialize. Defaults to None.
        appname (Optional[str], optional): The name of the application. Used if `directory`is None. Defaults to None.

    Returns:
        Union[str, PathLike[str]]: The directory.
    """
    if directory is None:
        directory = user_cache_dir(appname=appname)
        logging.debug(f"Directory defaulting to user-specific cache: {directory}")
    makedirs(directory, exist_ok=True)
    logging.debug(f"Directory created at: {directory}")
    return directory


def init_assets_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the assets directory.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (Optional[Union[str, PathLike[str]]], optional): The directory to initialize. Defaults to None.

    Returns:
        Union[str, PathLike[str]]: The directory.
    """
    return init_dir(directory, appname=ASSETS_CACHE_APPNAME)


def init_cached_assets_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the cached assets directory.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (Optional[Union[str, PathLike[str]]], optional): The directory to initialize. Defaults to None.

    Returns:
        Union[str, PathLike[str]]: The directory.
    """
    return init_dir(directory, appname=CACHED_ASSETS_CACHE_APPNAME)


def init_parquet_metadata_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the parquet metadata directory.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (Optional[Union[str, PathLike[str]]], optional): The directory to initialize. Defaults to None.

    Returns:
        Union[str, PathLike[str]]: The directory.
    """
    return init_dir(directory, appname=PARQUET_METADATA_CACHE_APPNAME)


def init_duckdb_index_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the duckdb index directory.

    If directory is None, it will be set to the default duckdb index location on the machine.

    Args:
        directory (Optional[Union[str, PathLike[str]]], optional): The directory to initialize. Defaults to None.

    Returns:
        Union[str, PathLike[str]]: The directory.
    """
    return init_dir(directory, appname=DUCKDB_INDEX_CACHE_APPNAME)


def exists(path: StrPath) -> bool:
    """Check if a path exists.

    Args:
        path (Union[str, PathLike[str]]): The path to check.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    return Path(path).exists()


def remove_dir(directory: StrPath) -> None:
    """Remove a directory.

    If the directory does not exist, don't raise.

    Args:
        directory (Union[str, PathLike[str]]): The directory to remove.
    """
    shutil.rmtree(directory, ignore_errors=True)
    logging.debug(f"Directory removed: {directory}")
