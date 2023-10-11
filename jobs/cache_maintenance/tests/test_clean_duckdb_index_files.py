# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import time
from pathlib import Path

from pytest import TempPathFactory

from cache_maintenance.clean_duckdb_index_files import clean_duckdb_index_files


def test_clean_duckdb_index_files(tmp_path_factory: TempPathFactory) -> None:
    duckdb_index_cache_directory = (
        tmp_path_factory.mktemp("test_clean_duckdb_index_files") / "duckdb_index_cache_directory"
    )
    duckdb_index_cache_directory.mkdir(parents=True, exist_ok=True)

    subdirectory = "downloads"
    file_extension = ".duckdb"
    expired_time_interval_seconds = 2
    os.mkdir(f"{duckdb_index_cache_directory}/{subdirectory}")

    index_file = Path(f"{duckdb_index_cache_directory}/{subdirectory}/index{file_extension}")
    index_file.touch()

    # ensure file exists
    assert index_file.is_file()

    # try to delete it inmediatly after creation, it should remain
    clean_duckdb_index_files(duckdb_index_cache_directory, subdirectory, expired_time_interval_seconds)
    assert index_file.is_file()

    # try to delete it after more that time interval, it should be deleted
    index_file.touch()
    time.sleep(expired_time_interval_seconds + 2)
    print(duckdb_index_cache_directory)
    print(subdirectory)
    clean_duckdb_index_files(duckdb_index_cache_directory, subdirectory, expired_time_interval_seconds)
    assert not index_file.is_file()

    os.rmdir(f"{duckdb_index_cache_directory}/{subdirectory}")
