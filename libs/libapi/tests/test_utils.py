# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import os
import time
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
from libcommon.storage import StrPath

from libapi.utils import clean_cached_assets


@pytest.mark.parametrize(
    "n_rows,keep_most_recent_rows_number,keep_first_rows_number,max_cleaned_rows_number,expected_remaining_rows",
    [
        (8, 1, 1, 100, [0, 7]),
        (8, 2, 2, 100, [0, 1, 6, 7]),
        (8, 1, 1, 3, [0, 4, 5, 6, 7]),
    ],
)
def test_clean_cached_assets(
    tmp_path: Path,
    n_rows: int,
    keep_most_recent_rows_number: int,
    keep_first_rows_number: int,
    max_cleaned_rows_number: int,
    expected_remaining_rows: list[int],
) -> None:
    cached_assets_directory = tmp_path / "cached-assets"
    split_dir = cached_assets_directory / "ds/--/plain_text/train"
    split_dir.mkdir(parents=True)
    for i in range(n_rows):
        (split_dir / str(i)).mkdir()
        time.sleep(0.01)

    def deterministic_glob_rows_in_assets_dir(
        dataset: str,
        assets_directory: StrPath,
    ) -> List[Path]:
        return sorted(
            list(Path(assets_directory).resolve().glob(os.path.join(dataset, "--", "*", "*", "*"))),
            key=lambda p: int(p.name),
        )

    with patch("libapi.utils.glob_rows_in_assets_dir", deterministic_glob_rows_in_assets_dir):
        clean_cached_assets(
            "ds",
            cached_assets_directory,
            keep_most_recent_rows_number=keep_most_recent_rows_number,
            keep_first_rows_number=keep_first_rows_number,
            max_cleaned_rows_number=max_cleaned_rows_number,
        )
    remaining_rows = sorted(int(row_dir.name) for row_dir in split_dir.glob("*"))
    assert remaining_rows == expected_remaining_rows
