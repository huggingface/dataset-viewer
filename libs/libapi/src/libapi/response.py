import logging

import pyarrow as pa
from datasets import Features
from libcommon.s3_client import S3Client
from libcommon.storage import StrPath
from libcommon.storage_options import S3StorageOptions
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE, PaginatedResponse
from libcommon.viewer_utils.features import to_features_list

from libapi.utils import to_rows_list

ROW_IDX_COLUMN = "__hf_index_id"


def create_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    s3_client: S3Client,
    cached_assets_s3_folder_name: str,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    num_rows_total: int,
    use_row_idx_column: bool = False,
) -> PaginatedResponse:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    logging.debug(f"create response for {dataset=} {config=} {split=}")
    storage_options = S3StorageOptions(
        assets_base_url=cached_assets_base_url,
        assets_directory=cached_assets_directory,
        overwrite=False,
        s3_client=s3_client,
        s3_folder_name=cached_assets_s3_folder_name,
    )
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(
            pa_table=pa_table,
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            storage_options=storage_options,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
            row_idx_column=ROW_IDX_COLUMN if use_row_idx_column else None,
        ),
        "num_rows_total": num_rows_total,
        "num_rows_per_page": MAX_NUM_ROWS_PER_PAGE,
    }
