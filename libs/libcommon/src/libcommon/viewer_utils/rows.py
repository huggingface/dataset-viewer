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
    """
    Create the response for the first rows of a split.

    The response contains the features, and the first rows of the split, obtained by calling get_rows_content.
    If the size of the response exceeds the maximum supported size or the maximum supported number of rows, the rows
    are "truncated" to fit within the restrictions:
    1. the last rows are removed, one by one, until the conditions are met.
    2. if the minimim number of rows has been reached, the remaining rows are "truncated" (backwards) to fit within the
         restrictions. For each row, the size of a each cell is computed as the size of its JSON serialization using
        orjson_dumps(). If a cell has less than min_cell_bytes, it is not truncated. If it has more, it is serialized
        to JSON and truncated to min_cell_bytes. The names of the truncated cells are listed in row_item["truncated_cells"].
        The cells from columns with type Audio or Image are never truncated (only applies to top-level types, nested audio
        files and images can be truncated).

    If the size of the response still exceeds the maximum supported size, a TooBigContentError is raised.

    Args:
        dataset (`str`): the dataset name.
        revision (`str`): the revision of the dataset.
        config (`str`): the config name.
        split (`str`): the split name.
        storage_client (`StorageClient`): the storage client to use to store the assets (audio, images).
        features (`Features`): the features to return in the response.
        get_rows_content (`GetRowsContent`): a callable that returns the rows content.
        min_cell_bytes (`int`): the minimum number of bytes for a cell, when truncation applies.
        rows_max_bytes (`int`): the maximum number of bytes for the response.
        rows_max_number (`int`): the maximum number of rows to return. The response will never contain more than this
            number of rows.
        rows_min_number (`int`): the minimum number of rows to return. The response will always contain at least this
            number of rows (provided that get_rows_content returns at least this number of rows).
        columns_max_number (`int`): the maximum number of columns to return. The response will never contain more than
            this number of columns.

    Raises:
        `TooBigContentError`: if the size of the content of the first rows exceeds the maximum supported size.
        `TooManyColumnsError`: if the number of columns exceeds the maximum supported number of columns.
        `RowsPostProcessingError`: if there is an error while post-processing the rows.

    Returns:
        [`SplitFirstRowsResponse`]: the response for the first rows of the split.
    """
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

    # return the response
    return response
