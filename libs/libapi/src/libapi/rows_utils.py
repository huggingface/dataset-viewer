# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from functools import partial
from typing import Any, Optional

import anyio
from datasets import Features
from libcommon.public_assets_storage import PublicAssetsStorage
from libcommon.utils import Row
from libcommon.viewer_utils.features import get_cell_value
from tqdm.contrib.concurrent import thread_map


def _transform_row(
    row_idx_and_row: tuple[int, Row],
    dataset: str,
    revision: str,
    config: str,
    split: str,
    features: Features,
    public_assets_storage: PublicAssetsStorage,
    offset: int,
    row_idx_column: Optional[str],
) -> Row:
    row_idx, row = row_idx_and_row
    transformed_row = {
        featureName: get_cell_value(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            row_idx=offset + row_idx if row_idx_column is None else row[row_idx_column],
            cell=row[featureName] if featureName in row else None,
            featureName=featureName,
            fieldType=fieldType,
            public_assets_storage=public_assets_storage,
        )
        for (featureName, fieldType) in features.items()
    }
    if row_idx_column and row_idx_column not in transformed_row:
        transformed_row |= {row_idx_column: row[row_idx_column]}
    return transformed_row


async def transform_rows(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    rows: list[Row],
    features: Features,
    public_assets_storage: PublicAssetsStorage,
    offset: int,
    row_idx_column: Optional[str],
) -> list[Row]:
    fn = partial(
        _transform_row,
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        features=features,
        public_assets_storage=public_assets_storage,
        offset=offset,
        row_idx_column=row_idx_column,
    )
    if "Audio(" in str(features) or "Image(" in str(features):
        # Use multithreading to parallelize image/audio files uploads.
        # Also multithreading is ok to convert audio data
        # (we use pydub which might spawn one ffmpeg process per conversion, which releases the GIL)
        desc = f"_transform_row for {dataset}"
        _thread_map = partial(thread_map, desc=desc, total=len(rows))
        return await anyio.to_thread.run_sync(_thread_map, fn, enumerate(rows))
    else:

        def _map(func: Callable[[Any], Any], *iterables: Any) -> list[Row]:
            return list(map(func, *iterables))

        return await anyio.to_thread.run_sync(_map, fn, enumerate(rows))
