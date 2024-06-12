# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import pytest

from worker.utils import get_file_extension


@pytest.mark.parametrize(
    "filename,expected_extension",
    [
        ("README.md", ".md"),
        ("file.csv", ".csv"),
        # leading dots are ignored
        (".gitattributes", ""),
        (".file.csv", ".csv"),
        ("....file.csv", ".csv"),
        # no extension
        ("LICENSE", ""),
        # multiple dots
        ("file.with.dots.csv", ".csv"),
        # special case for tar
        ("file.tar.gz", ".tar.gz"),
        ("file.with.dots.tar.gz", ".tar.gz"),
        ("file.tar.bz2", ".tar.bz2"),
        ("file.tar.unknown", ".unknown"),
        ("file.tar", ".tar"),
        ("file.nottar.gz", ".gz"),
        # clean suffixes
        ("file.csv?dl=1", ".csv"),
        ("file.csv_1", ".csv"),
        ("file.csv-00000-of-00001", ".csv"),
        # ignore paths
        ("path/to/file.csv", ".csv"),
        (".path/to.some/file.csv", ".csv"),
        ("path/to/.gitignore", ""),
    ],
)
def test_get_file_extension(filename: str, expected_extension: str) -> None:
    assert get_file_extension(filename) == expected_extension
