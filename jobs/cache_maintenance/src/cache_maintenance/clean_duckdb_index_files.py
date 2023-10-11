# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import glob
import logging
import os
from datetime import datetime, timedelta

from libcommon.storage import StrPath


def clean_duckdb_index_files(
    duckdb_index_cache_directory: StrPath, subdirectory: str, expired_time_interval_seconds: int
) -> None:
    """
    Clean obsolete files after expired_time_interval_seconds in duckdb_index_cache_directory
    """
    logging.info("clean duckdb files")
    folder_pattern = f"{duckdb_index_cache_directory}/{subdirectory}/*"
    logging.info(f"looking for all files with pattern {folder_pattern}")
    now = datetime.now().replace(tzinfo=None)
    for path in glob.glob(folder_pattern, recursive=True):
        last_access_time_value = os.path.getatime(path)
        last_access_datetime = datetime.fromtimestamp(last_access_time_value).replace(tzinfo=None)
        if last_access_datetime + timedelta(seconds=expired_time_interval_seconds) <= now:
            logging.info(f"deleting file {path=} {last_access_datetime=}")
            os.remove(path)
