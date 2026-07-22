# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import get_dataset_config_info, get_dataset_split_names
from datasets.packaged_modules.arrow.arrow import Arrow
from datasets.packaged_modules.csv.csv import Csv
from libcommon.exceptions import DatasetWithArrowFilesNotSupportedError

from worker.utils import (
    FileExtension,
    get_file_extension,
    get_rows_or_raise,
    safe_inspect,
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


def test_safe_load_dataset_builder_rejects_arrow_builder() -> None:
    with (
        patch("datasets.load.dataset_module_factory", return_value=_get_dataset_module()),
        patch("datasets.load.get_dataset_builder_class", return_value=Arrow),
        pytest.raises(DatasetWithArrowFilesNotSupportedError) as error_info,
    ):
        safe_load_dataset_builder(path="namespace/dataset", name="default")

    assert error_info.value.code == "DatasetWithArrowFilesNotSupportedError"
    assert error_info.value.status_code == HTTPStatus.NOT_IMPLEMENTED


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


def test_safe_inspect_rejects_arrow_builder_in_get_dataset_config_info() -> None:
    with (
        patch("datasets.load.dataset_module_factory", return_value=_get_dataset_module()),
        patch("datasets.load.get_dataset_builder_class", return_value=Arrow),
        safe_inspect,
        pytest.raises(DatasetWithArrowFilesNotSupportedError),
    ):
        get_dataset_config_info(path="namespace/dataset", config_name="default")


def test_safe_inspect_rejects_arrow_builder_in_get_dataset_split_names() -> None:
    with (
        patch("datasets.load.dataset_module_factory", return_value=_get_dataset_module()),
        patch("datasets.load.get_dataset_builder_class", return_value=Arrow),
        safe_inspect,
        pytest.raises(DatasetWithArrowFilesNotSupportedError),
    ):
        get_dataset_split_names(path="namespace/dataset", config_name="default")


def test_get_rows_or_raise_preserves_arrow_files_error() -> None:
    with (
        patch("worker.utils.get_rows", side_effect=DatasetWithArrowFilesNotSupportedError()),
        pytest.raises(DatasetWithArrowFilesNotSupportedError),
    ):
        get_rows_or_raise(dataset="namespace/dataset", config="default", split="train", rows_max_number=10, token=None)
