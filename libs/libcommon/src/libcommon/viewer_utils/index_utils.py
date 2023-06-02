# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from os import makedirs
from pathlib import Path
from libcommon.storage import StrPath
from typing import Tuple

DATASET_SEPARATOR = "--"
INDEX_DIR_MODE = 0o755


def create_index_dir_split(
    dataset: str, config: str, split: str, index_directory: StrPath
) -> Tuple[str, str]:
    split_path = dataset / DATASET_SEPARATOR / config / split
    dir_path = Path(index_directory).resolve() / split_path
    makedirs(dir_path, INDEX_DIR_MODE, exist_ok=True)
    return split_path, dir_path

