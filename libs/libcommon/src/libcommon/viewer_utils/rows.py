# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from typing import Protocol

from datasets import Audio, Features, Image

from libcommon.dtos import Row, RowItem, RowsContent, SplitFirstRowsResponse
from libcommon.exceptions import (
    RowsPostProcessingError,
    TooBigContentError,
    TooManyColumnsError,
)
from libcommon.storage_client import StorageClient
from libcommon.utils import SmallerThanMaxBytesError, get_json_size, serialize_and_truncate
from libcommon.viewer_utils.features import get_cell_value, to_features_list


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


def to_row_item(row_idx: int, row: Row) -> RowItem:
    return {
        "row_idx": row_idx,
        "row": row,
        "truncated_cells": [],
    }


def truncate_row_item(row_item: RowItem, min_cell_bytes: int, columns_to_keep_untruncated: list[str]) -> RowItem:
    """
    Truncate all the cells of a row item to min_cell_bytes, and return the row item.

    The row item is mutated, and the cells are replaced by their JSON serialization, truncated to min_cell_bytes.
    The names of the truncated cells are listed in row_item["truncated_cells"].

    Args:
        row_item ([`RowItem`]): the row item to truncate
        min_cell_bytes (`int`): the minimum number of bytes for a cell. If a cell has less than this number of bytes,
            it is not truncated. If it has more, it is truncated to this number of bytes.
            The size of a cell is computed as the size of its JSON serialization using orjson_dumps().
        columns_to_keep_untruncated (`list[str]`): the list of columns to keep untruncated

    Returns:
        [`RowItem`]: the same row item, mutated, with all the cells truncated to min_cell_bytes
    """
    for column_name, cell in row_item["row"].items():
        if column_name in columns_to_keep_untruncated:
            # we keep the cell untouched
            continue
        try:
            truncated_serialized_cell = serialize_and_truncate(obj=cell, max_bytes=min_cell_bytes)
            row_item["row"][column_name] = truncated_serialized_cell
            if column_name not in row_item["truncated_cells"]:
                row_item["truncated_cells"].append(column_name)
        except SmallerThanMaxBytesError:
            # the cell serialization is smaller than min_cell_bytes, we keep it untouched
            continue
    return row_item


COMMA_SIZE = 1  # the comma "," is encoded with one byte in utf-8


# Mutates row_items, and returns them anyway
def truncate_row_items(
    row_items: list[RowItem], min_cell_bytes: int, rows_max_bytes: int, columns_to_keep_untruncated: list[str]
) -> list[RowItem]:
    # compute the current size
    rows_bytes = sum(get_json_size(row_item) for row_item in row_items) + COMMA_SIZE * (len(row_items) - 1)

    # Loop backwards, so that the last rows are truncated first
    for row_item in reversed(row_items):
        if rows_bytes < rows_max_bytes:
            break
        previous_size = get_json_size(row_item) + COMMA_SIZE
        row_item = truncate_row_item(
            row_item=row_item, min_cell_bytes=min_cell_bytes, columns_to_keep_untruncated=columns_to_keep_untruncated
        )
        new_size = get_json_size(row_item) + COMMA_SIZE
        rows_bytes += new_size - previous_size
    return row_items


def create_truncated_row_items(
    rows: list[Row],
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_min_number: int,
    columns_to_keep_untruncated: list[str],
) -> tuple[list[RowItem], bool]:
    row_items = []
    rows_bytes = 0

    # two restrictions must be enforced:
    # - at least rows_min_number rows
    # - at most rows_max_bytes bytes. Note that it's the limit to the sum of the rows sizes. The JSON response size
    #   will be greater, due to the other fields (row_idx, truncated_cells, features, etc.).
    # To enforce this:
    # 1. first get the first rows_min_number rows
    for row_idx, row in enumerate(rows[:rows_min_number]):
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        row_items.append(row_item)

    # 2. if the total is over the bytes limit, truncate the values, iterating backwards starting
    # from the last rows, until getting under the threshold
    # caveat: the truncation might not be enough to get under the threshold if:
    # - the number of columns is too high
    # - rows_max_bytes is too low (or even negative)
    if rows_bytes >= rows_max_bytes:
        # logging.debug(
        #     f"the size of the first {rows_min_number} rows ({rows_bytes}) is above the max number of bytes"
        #     f" ({rows_max_bytes}), they will be truncated"
        # )
        truncated_row_items = truncate_row_items(
            row_items=row_items,
            min_cell_bytes=min_cell_bytes,
            rows_max_bytes=rows_max_bytes,
            columns_to_keep_untruncated=columns_to_keep_untruncated,
        )
        return truncated_row_items, len(truncated_row_items) < len(rows)

    # 3. else: add the remaining rows until the end, or until the bytes threshold
    for idx, row in enumerate(rows[rows_min_number:]):
        row_idx = rows_min_number + idx
        row_item = to_row_item(row_idx=row_idx, row=row)
        rows_bytes += get_json_size(row_item) + COMMA_SIZE
        if rows_bytes >= rows_max_bytes:
            # logging.debug(
            #     f"the rows in the split have been truncated to {row_idx} row(s) to keep the size"
            #     f" ({rows_bytes}) under the limit ({rows_max_bytes})"
            # )
            break
        row_items.append(row_item)
    return row_items, len(row_items) < len(rows)


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

    # return the response
    return response
