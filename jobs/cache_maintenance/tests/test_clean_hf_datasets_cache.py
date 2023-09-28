# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import time
from pathlib import Path

from pytest import TempPathFactory

from cache_maintenance.clean_hf_datasets_cache import clean_hf_datasets_cache


def test_clean_hf_datasets_cache(tmp_path_factory: TempPathFactory) -> None:
    hf_datasets_cache_directory = str(tmp_path_factory.mktemp("test_clean_hf_datasets_cache") / "hf_datasets_cache")
    expired_time_interval_seconds = 2
    os.mkdir(hf_datasets_cache_directory)

    cache_file = Path(f"{hf_datasets_cache_directory}/foo")
    cache_file.touch()

    # ensure file exists
    assert cache_file.is_file()

    # try to delete it inmediatly after creation, it should remain
    clean_hf_datasets_cache(hf_datasets_cache_directory, expired_time_interval_seconds)
    assert cache_file.is_file()

    # try to delete it after more that time interval, it should be deleted
    cache_file.touch()
    time.sleep(expired_time_interval_seconds + 2)
    clean_hf_datasets_cache(hf_datasets_cache_directory, expired_time_interval_seconds)
    assert not cache_file.is_file()

    os.rmdir(f"{hf_datasets_cache_directory}")
