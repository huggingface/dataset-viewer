# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from typing import Protocol

from datasets import Audio, Features, Image

from libcommon.dtos import Row, RowsContent, SplitFirstRowsResponse
from libcommon.exceptions import (
    RowsPostProcessingError,
    TooBigContentError,
    TooManyColumnsError,
)
from libcommon.storage_client import StorageClient
from libcommon.utils import get_json_size
from libcommon.viewer_utils.features import get_cell_value, to_features_list
from libcommon.viewer_utils.truncate_rows import create_truncated_row_items


def transform_rows(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    rows: list[Row],
    features: Features,
    storage_client: StorageClient,
) -> list[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=row[featureName] if featureName in row else None,
                featureName=featureName,
                fieldType=fieldType,
                storage_client=storage_client,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


class GetRowsContent(Protocol):
    def __call__(self, rows_max_number: int) -> RowsContent:
        ...


def create_first_rows_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    storage_client: StorageClient,
    features: Features,
    get_rows_content: GetRowsContent,
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
) -> SplitFirstRowsResponse:
    if features and len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(features)}) exceeds the maximum supported number of columns"
            f" ({columns_max_number}). This is a current limitation of the datasets viewer. You can reduce the number"
            " of columns if you want the viewer to work."
        )

    # validate size of response without the rows
    features_list = to_features_list(features=features)
    response_features_only: SplitFirstRowsResponse = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": features_list,
        "rows": [],
        "truncated": False,
    }

    surrounding_json_size = get_json_size(response_features_only)
    if surrounding_json_size > rows_max_bytes:
        raise TooBigContentError(
            f"The size of the content of the first rows ({surrounding_json_size} B) exceeds the maximum"
            f" supported size ({rows_max_bytes} B) even after truncation. Please report the issue."
        )

    rows_content = get_rows_content(rows_max_number)
    if len(rows_content.rows) > rows_max_number:
        raise ValueError(
            f"The number of rows ({len(rows_content.rows)}) exceeds the maximum supported number of rows"
            f" ({rows_max_number})."
        )

    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            rows=rows_content.rows,
            features=features,
            storage_client=storage_client,
        )
    except Exception as err:
        raise RowsPostProcessingError(
            "Server error while post-processing the split rows. Please report the issue.",
            cause=err,
        ) from err

    # truncate the rows to fit within the restrictions, and prepare them as RowItems
    columns_to_keep_untruncated = [col for col, feature in features.items() if isinstance(feature, (Image, Audio))]
    row_items, truncated = create_truncated_row_items(
        rows=transformed_rows,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes - surrounding_json_size,
        rows_min_number=rows_min_number,
        columns_to_keep_untruncated=columns_to_keep_untruncated,
    )

    response = response_features_only
    response["rows"] = row_items
    response["truncated"] = (not rows_content.all_fetched) or truncated
    # ^ this boolean contains two different infos:
    #   - if the dataset contains more than rows_max_number rows, or
    #   - if some rows have been deleted OR some cells have been serialized+truncated, to fit the budget of rows_max_bytes
    # The only way for it to be True is if the dataset contains less than rows_max_number rows, and if their size is
    # small enough to fit within rows_max_bytes. It only happens for very small datasets.
    # Note that changing it means recomputing all the /first-rows responses.

    # return the response
    return response
