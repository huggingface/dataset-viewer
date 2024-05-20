import logging
from typing import Optional

import pyarrow as pa
from datasets import Features
from libcommon.constants import MAX_NUM_ROWS_PER_PAGE, ROW_IDX_COLUMN
from libcommon.dtos import PaginatedResponse
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.features import to_features_list

from libapi.utils import to_rows_list


async def create_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    storage_client: StorageClient,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    num_rows_total: int,
    partial: bool,
    use_row_idx_column: bool = False,
    truncated_columns: Optional[list[str]] = None,
) -> PaginatedResponse:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    logging.debug(f"create response for {dataset=} {config=} {split=}")
    return {
        "features": [
            feature_item
            for feature_item in to_features_list(features)
            if not use_row_idx_column or feature_item["name"] != ROW_IDX_COLUMN
        ],
        "rows": await to_rows_list(
            pa_table=pa_table,
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            storage_client=storage_client,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
            row_idx_column=ROW_IDX_COLUMN if use_row_idx_column else None,
            truncated_columns=truncated_columns,
        ),
        "num_rows_total": num_rows_total,
        "num_rows_per_page": MAX_NUM_ROWS_PER_PAGE,
        "partial": partial,
    }
