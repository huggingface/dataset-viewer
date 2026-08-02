# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from pathlib import Path

import datasets
import lance
import lance.file
import pyarrow as pa
import pytest


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
