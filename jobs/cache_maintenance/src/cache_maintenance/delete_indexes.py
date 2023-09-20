# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import glob
import logging
import os
from datetime import datetime, timedelta

from libcommon.storage import StrPath


def delete_indexes(
    duckdb_index_cache_directory: StrPath, subdirectory: str, file_extension: str, expired_time_interval_seconds: int
) -> None:
    """
    Delete temporary DuckDB index files downloaded to handle /search requests
    """
    logging.info("delete indexes")
    indexes_folder = f"{duckdb_index_cache_directory}/{subdirectory}/**/*{file_extension}"
    logging.info(f"looking for all files with pattern {indexes_folder}")
    now = datetime.now().replace(tzinfo=None)
    for path in glob.glob(indexes_folder, recursive=True):
        last_access_time_value = os.path.getatime(path)
        last_access_datetime = datetime.fromtimestamp(last_access_time_value).replace(tzinfo=None)
        if last_access_datetime + timedelta(seconds=expired_time_interval_seconds) <= now:
            logging.info(f"deleting file {path=} {last_access_datetime=}")
            os.remove(path)
