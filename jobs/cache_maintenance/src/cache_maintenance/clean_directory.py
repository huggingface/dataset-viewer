# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import glob
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Any


def clean_directory(pattern: str, expired_time_interval_seconds: int) -> None:
    """
    Delete temporary cache directories that match the given pattern.
    """
    # sanity check
    if len(str(pattern)) < 10:
        raise RuntimeError(f"Sanity check on pattern failed: len('{pattern}') < 10.")
    logging.info(f"looking for all files and directories with pattern {pattern}")
    now = datetime.now().replace(tzinfo=None)
    errors = 0
    total_dirs = 0
    total_files = 0

    def rmtree_on_error(function: Any, path: str, excinfo: Any) -> None:  # noqa: ARG001, args needed for onerror=
        logging.error(f"failed to delete directory {path=}")

    for path in glob.glob(pattern):
        last_access_time_value = os.path.getatime(path)
        last_access_datetime = datetime.fromtimestamp(last_access_time_value).replace(tzinfo=None)
        if last_access_datetime + timedelta(seconds=expired_time_interval_seconds) <= now:
            if os.path.isfile(path):
                logging.info(f"deleting file {path=} {last_access_datetime=}")
                os.remove(path)
                total_files += 1
            elif os.path.isdir(path):
                logging.info(f"deleting directory {path=} {last_access_datetime=}")
                shutil.rmtree(path, onerror=rmtree_on_error)
                total_dirs += 1
                errors += os.path.isdir(path)
    if errors:
        logging.error(f"clean_directory failed to remove {errors} directories at the root of the cache directory.")
    if total_files:
        logging.info(f"clean_directory removed {total_files} files at the root of the cache directory.")

    logging.info(f"clean_directory removed {total_dirs - errors} directories at the root of the cache directory.")
