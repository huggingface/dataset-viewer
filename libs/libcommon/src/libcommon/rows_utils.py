# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from typing import List

from datasets import Features

from libcommon.storage import StrPath
from libcommon.utils import Row
from libcommon.viewer_utils.features import get_cell_value


def transform_rows(
    dataset: str,
    config: str,
    split: str,
    rows: List[Row],
    features: Features,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
) -> List[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=offset + row_idx,
                cell=row[featureName] if featureName in row else None,
                featureName=featureName,
                fieldType=fieldType,
                assets_base_url=cached_assets_base_url,
                assets_directory=cached_assets_directory,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]
