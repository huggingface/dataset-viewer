# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import time
from pathlib import Path
from typing import Optional

import pytest
from pytest import TempPathFactory

from libcommon.storage import StrPath, clean_dir, init_dir, remove_dir


@pytest.mark.parametrize(
    "has_directory,is_directory_string,has_appname",
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_init_dir(
    tmp_path_factory: pytest.TempPathFactory, has_directory: bool, is_directory_string: bool, has_appname: bool
) -> None:
    subdirectory = "subdirectory"
    tmp_path = tmp_path_factory.mktemp("test") / subdirectory
    appname = "appname" if has_appname else None
    directory: Optional[StrPath]
    if has_directory:
        directory = str(tmp_path) if is_directory_string else tmp_path
        result = init_dir(directory=directory, appname=appname)
        assert result == directory
        assert subdirectory in str(result), result
        if appname is not None:
            assert appname not in str(result), result
    else:
        directory = None
        result = init_dir(directory=directory, appname=appname)
        assert result != directory, result
        assert subdirectory not in str(result), result
        assert isinstance(result, str), result
        if appname:
            assert appname in str(result), result
    Path(result).exists()
    Path(result).is_dir()


@pytest.mark.parametrize(
    "exists,is_string",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_remove_dir(tmp_path_factory: pytest.TempPathFactory, exists: bool, is_string: bool) -> None:
    subdirectory = "subdirectory"
    tmp_path = tmp_path_factory.mktemp("test") / subdirectory
    tmp_file = tmp_path / "file.txt"
    if exists:
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_file.touch()
    assert tmp_path.exists() is exists
    assert tmp_path.is_dir() is exists
    assert tmp_file.exists() is exists
    assert tmp_file.is_file() is exists
    directory: StrPath = str(tmp_path) if is_string else tmp_path
    remove_dir(directory)
    assert not tmp_path.exists()
    assert not tmp_path.is_dir()
    assert not tmp_file.exists()
    assert not tmp_file.is_file()


def test_clean_directory(tmp_path_factory: TempPathFactory) -> None:
    root_folder = tmp_path_factory.mktemp("test_clean_directory") / "root_folder"
    root_folder.mkdir(parents=True, exist_ok=True)
    expired_time_interval_seconds = 2

    test_dataset_cache = root_folder / "medium/datasets/test_dataset_cache"
    test_dataset_cache.mkdir(parents=True, exist_ok=True)
    cache_file = test_dataset_cache / "foo.txt"
    cache_file.touch()

    # ensure file exists
    assert cache_file.is_file()
    # try to delete it inmediatly after creation, it should remain
    clean_dir(root_folder, expired_time_interval_seconds)
    assert cache_file.is_file()
    assert test_dataset_cache.is_dir()

    # try to delete it after more that time interval, it should be deleted
    cache_file.touch()
    time.sleep(expired_time_interval_seconds + 2)
    clean_dir(root_folder, expired_time_interval_seconds)
    assert not cache_file.is_file()
    assert not test_dataset_cache.is_dir()  # it should be deleted because is empty
