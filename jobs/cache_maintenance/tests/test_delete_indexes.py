# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import time
from pathlib import Path

from cache_maintenance.delete_indexes import delete_indexes


def test_delete_indexes() -> None:
    duckdb_index_cache_directory = "/tmp"
    subdirectory = "download"
    file_extension = ".duckdb"
    expired_time_interval_seconds = 2
    os.mkdir(f"{duckdb_index_cache_directory}/{subdirectory}")

    index_file = Path(f"{duckdb_index_cache_directory}/{subdirectory}/index{file_extension}")
    index_file.touch()

    # ensure file exists
    assert index_file.is_file()

    # try to delete it inmediatly after creation, it should remain
    delete_indexes(duckdb_index_cache_directory, subdirectory, file_extension, expired_time_interval_seconds)
    assert index_file.is_file()

    # try to delete it after more that time interval, it should be deleted
    index_file.touch()
    time.sleep(expired_time_interval_seconds + 2)
    delete_indexes(duckdb_index_cache_directory, subdirectory, file_extension, expired_time_interval_seconds)
    assert not index_file.is_file()

    os.rmdir(f"{duckdb_index_cache_directory}/{subdirectory}")
