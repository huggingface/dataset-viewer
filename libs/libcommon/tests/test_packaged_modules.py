# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import subprocess
import sys
from pathlib import Path
from typing import Union

import datasets
import lance
import lance.file
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from libcommon.exceptions import ParquetBrotliCompressionNotSupportedError


def test_libcommon_imports_without_libviewer() -> None:
    # the api, admin, sse-api and webhook images ship without libviewer, and importing libcommon applies the patches
    subprocess.run([sys.executable, "-c", "import sys; sys.modules['libviewer'] = None; import libcommon"], check=True)


@pytest.mark.parametrize("as_dataset_dir", [True, False])
def test_lance_data_is_refused(tmp_path: Path, as_dataset_dir: bool) -> None:
    table = pa.table({"category": pa.array([b"alpha", b"beta", b"gamma"], type=pa.binary())})
    if as_dataset_dir:
        lance.write_dataset(table, str(tmp_path / "data.lance"))
    else:
        with lance.file.LanceFileWriter(str(tmp_path / "data.lance")) as writer:
            writer.write_batch(table)

    with pytest.raises(NotImplementedError, match="Lance format is not supported"):
        datasets.load_dataset(str(tmp_path), split="train")


def write_parquet(table: pa.Table, path: Path, compression: Union[str, dict[str, str]], refused: bool) -> None:
    pq.write_table(table, path, compression=compression)  # type: ignore
    if refused:
        # Zero the pages but keep the footer, so that any page read fails with another error
        data = bytearray(path.read_bytes())
        footer_start = len(data) - 8 - int.from_bytes(data[-8:-4], "little")
        data[4:footer_start] = bytes(footer_start - 4)
        path.write_bytes(data)


@pytest.mark.parametrize(
    "compression,refused",
    [
        ("brotli", True),
        ({"text": "snappy", "label": "brotli"}, True),
        ("snappy", False),
        ("none", False),
    ],
)
def test_brotli_parquet_data_is_refused(
    tmp_path: Path, compression: Union[str, dict[str, str]], refused: bool
) -> None:
    table = pa.table({"text": ["Hello there", "General Kenobi"], "label": [0, 1]})
    write_parquet(table, tmp_path / "data.parquet", compression=compression, refused=refused)

    if refused:
        with pytest.raises(ParquetBrotliCompressionNotSupportedError):
            list(datasets.load_dataset(str(tmp_path), split="train", streaming=True))
    else:
        assert list(datasets.load_dataset(str(tmp_path), split="train", streaming=True)) == table.to_pylist()


@pytest.mark.parametrize("compression,refused", [("brotli", True), ("snappy", False)])
def test_brotli_metadata_parquet_is_refused(tmp_path: Path, compression: str, refused: bool) -> None:
    Image.new("RGB", (1, 1)).save(tmp_path / "image.png")
    metadata = pa.table({"file_name": ["image.png"], "caption": ["a pixel"]})
    write_parquet(metadata, tmp_path / "metadata.parquet", compression=compression, refused=refused)

    if refused:
        with pytest.raises(ParquetBrotliCompressionNotSupportedError):
            list(datasets.load_dataset(str(tmp_path), split="train", streaming=True))
    else:
        rows = list(datasets.load_dataset(str(tmp_path), split="train", streaming=True))
        assert [row["caption"] for row in rows] == ["a pixel"]
