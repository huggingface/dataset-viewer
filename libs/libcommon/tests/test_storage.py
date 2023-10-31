# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path
from typing import Optional

import pytest

from libcommon.storage import StrPath, init_dir, remove_dir


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
        assert type(result) is str, result
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
