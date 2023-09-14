# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from functools import partial
from typing import Optional

from datasets import Features
from tqdm.contrib.concurrent import thread_map

from libcommon.storage import StrPath
from libcommon.utils import Row
from libcommon.viewer_utils.features import get_cell_value


def _transform_row(
    row_idx_and_row: tuple[int, Row],
    dataset: str,
    config: str,
    split: str,
    features: Features,
    assets_base_url: str,
    assets_directory: StrPath,
    offset: int,
    row_idx_column: Optional[str],
) -> Row:
    row_idx, row = row_idx_and_row
    return {
        featureName: get_cell_value(
            dataset=dataset,
            config=config,
            split=split,
            row_idx=offset + row_idx if row_idx_column is None else row[row_idx_column],
            cell=row[featureName] if featureName in row else None,
            featureName=featureName,
            fieldType=fieldType,
            assets_base_url=assets_base_url,
            assets_directory=assets_directory,
        )
        for (featureName, fieldType) in features.items()
    }


def transform_rows(
    dataset: str,
    config: str,
    split: str,
    rows: list[Row],
    features: Features,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
    row_idx_column: Optional[str],
) -> list[Row]:
    fn = partial(
        _transform_row,
        dataset=dataset,
        config=config,
        split=split,
        features=features,
        assets_base_url=cached_assets_base_url,
        assets_directory=cached_assets_directory,
        offset=offset,
        row_idx_column=row_idx_column,
    )
    if "Audio(" in str(features):
        # use multithreading to parallelize audio files processing
        # (we use pydub which might spawn one ffmpeg process per conversion, which releases the GIL)
        desc = f"transform_rows(audio) for {dataset}"
        return thread_map(fn, enumerate(rows), desc=desc, total=len(rows))  # type: ignore
    else:
        return [fn((row_idx, row)) for row_idx, row in enumerate(rows)]
