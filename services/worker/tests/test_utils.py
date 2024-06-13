# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


import pytest

from worker.utils import FileExtension, get_file_extension


@pytest.mark.parametrize(
    "filename,expected_extension",
    [
        ("README.md", FileExtension(extension=".md")),
        ("file.csv", FileExtension(extension=".csv")),
        # leading dots are ignored
        (".gitattributes", FileExtension(extension="")),
        (".file.csv", FileExtension(extension=".csv")),
        ("....file.csv", FileExtension(extension=".csv")),
        # no extension
        ("LICENSE", FileExtension(extension="")),
        # multiple dots
        ("file.with.dots.csv", FileExtension(extension=".csv")),
        # clean suffixes
        ("file.csv?dl=1", FileExtension(extension=".csv")),
        ("file.csv_1", FileExtension(extension=".csv")),
        ("file.csv-00000-of-00001", FileExtension(extension=".csv")),
        # ignore paths
        ("path/to/file.csv", FileExtension(extension=".csv")),
        (".path/to.some/file.csv", FileExtension(extension=".csv")),
        ("path/to/.gitignore", FileExtension(extension="")),
        # double extensions
        ("file.tar.gz", FileExtension(extension=".gz", uncompressed_extension=".tar")),
        ("file.with.dots.tar.gz", FileExtension(extension=".gz", uncompressed_extension=".tar")),
        ("file.tar.bz2", FileExtension(extension=".bz2", uncompressed_extension=".tar")),
        ("file.jsonl.gz", FileExtension(extension=".gz", uncompressed_extension=".jsonl")),
        ("file.tar.unknown", FileExtension(extension=".unknown")),
        ("file.tar", FileExtension(extension=".tar")),
    ],
)
def test_get_file_extension(filename: str, expected_extension: FileExtension) -> None:
    assert get_file_extension(filename).extension == expected_extension.extension
    assert get_file_extension(filename).uncompressed_extension == expected_extension.uncompressed_extension
