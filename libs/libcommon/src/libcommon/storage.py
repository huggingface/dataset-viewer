# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import shutil
from os import PathLike, makedirs
from typing import Optional, Union

from appdirs import user_cache_dir  # type:ignore

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


def empty_dir(directory: StrPath) -> None:
    """Empty a directory.

    If the directory does not exist, it will be created.

    Args:
        directory (Union[str, PathLike[str]]): The directory to empty.
    """
    shutil.rmtree(directory, ignore_errors=True)
    makedirs(directory, exist_ok=True)
    logging.debug(f"Directory emptied: {directory}")
