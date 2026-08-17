# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from types import SimpleNamespace
from unittest.mock import patch

import datasets.data_files
import pytest
from datasets.packaged_modules.csv.csv import Csv

from worker.utils import (
    FileExtension,
    allow_only_relative_data_files,
    get_file_extension,
    safe_load_dataset_builder,
)


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
        # case insensitive
        ("file.CSV", FileExtension(extension=".csv")),
        ("file.CSv", FileExtension(extension=".csv")),
        ("file.CSV?dl=1", FileExtension(extension=".csv")),
        ("file.with.dots.TAR.GZ", FileExtension(extension=".gz", uncompressed_extension=".tar")),
    ],
)
def test_get_file_extension(filename: str, expected_extension: FileExtension) -> None:
    assert get_file_extension(filename).extension == expected_extension.extension
    assert get_file_extension(filename).uncompressed_extension == expected_extension.uncompressed_extension


def _get_dataset_module() -> SimpleNamespace:
    return SimpleNamespace(
        builder_configs_parameters=SimpleNamespace(default_config_name="default"),
        builder_kwargs={"base_path": "hf://datasets/namespace/dataset@revision"},
        dataset_infos={},
        hash="revision",
    )


def test_safe_load_dataset_builder_allows_non_arrow_builder() -> None:
    class CsvBuilder(Csv):  # type: ignore[misc]
        builder_configs = {"default": SimpleNamespace(data_files=None)}

        def __init__(self, **kwargs: object) -> None:
            pass

    with (
        patch("datasets.load.dataset_module_factory", return_value=_get_dataset_module()),
        patch("datasets.load.get_dataset_builder_class", return_value=CsvBuilder),
    ):
        # data_files=None and download_mode=None mirror what `datasets.inspect` passes
        builder = safe_load_dataset_builder(
            path="namespace/dataset", name="default", data_files=None, download_mode=None
        )

    assert isinstance(builder, CsvBuilder)


@pytest.mark.parametrize(
    "pattern",
    [
        "http://169.254.169.254/latest/meta-data/",
        "https://example.org/data.csv",
        "s3://bucket/data.csv",
        "/etc/passwd",
    ],
)
def test_allow_only_relative_data_files_refuses_a_pattern_outside_of_the_repository(pattern: str) -> None:
    with patch("worker.utils.resolve_pattern") as resolver, allow_only_relative_data_files():
        with pytest.raises(ValueError, match="Data files don't belong to"):
            datasets.data_files.resolve_pattern(pattern, base_path="hf://datasets/namespace/dataset@revision")
    # `resolve_pattern` sends a request before it looks the protocol up, so it must not be reached
    resolver.assert_not_called()


def test_safe_load_dataset_builder_guards_the_data_files_resolution() -> None:
    class CsvBuilder(Csv):  # type: ignore[misc]
        builder_configs = {"default": SimpleNamespace(data_files=None)}

        def __init__(self, **kwargs: object) -> None:
            pass

    def dataset_module_factory(*args: object, **kwargs: object) -> SimpleNamespace:  # noqa: ARG001
        # `datasets` resolves the data files patterns from here, and resolving is what sends the
        # request, so the guard has to be active by now. This covers every job runner that reaches
        # `dataset_module_factory` through `safe_load_dataset_builder`: config-split-names,
        # split-first-rows and config-parquet-and-info.
        with pytest.raises(ValueError, match="Data files don't belong to"):
            datasets.data_files.resolve_pattern(
                "https://example.org/data.csv", base_path="hf://datasets/namespace/dataset@revision"
            )
        return _get_dataset_module()

    with (
        patch("datasets.load.dataset_module_factory", dataset_module_factory),
        patch("datasets.load.get_dataset_builder_class", return_value=CsvBuilder),
    ):
        builder = safe_load_dataset_builder(
            path="namespace/dataset", name="default", data_files=None, download_mode=None
        )

    assert isinstance(builder, CsvBuilder)


def test_allow_only_relative_data_files_allows_a_relative_pattern() -> None:
    resolved = ["hf://datasets/namespace/dataset@revision/data/train.csv"]
    with allow_only_relative_data_files(), patch("worker.utils.resolve_pattern", return_value=resolved) as resolver:
        assert (
            datasets.data_files.resolve_pattern("data/*.csv", base_path="hf://datasets/namespace/dataset@revision")
            == resolved
        )
    resolver.assert_called_once()
