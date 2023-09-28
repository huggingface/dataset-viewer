import pyarrow as pa
from datasets import Features
from libcommon.s3_client import S3Client
from libcommon.storage import StrPath
from libcommon.storage_options import DirectoryStorageOptions, S3StorageOptions
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE, PaginatedResponse
from libcommon.viewer_utils.features import to_features_list

from libapi.utils import to_rows_list

CACHED_ASSETS_S3_SUPPORTED_DATASETS: list[str] = ["asoria/image"]  # for testing


def create_response(
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    s3_client: S3Client,
    cached_assets_s3_bucket: str,
    cached_assets_s3_folder_name: str,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    num_rows_total: int,
) -> PaginatedResponse:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    use_s3_storage = dataset in CACHED_ASSETS_S3_SUPPORTED_DATASETS
    storage_options = (
        S3StorageOptions(
            assets_base_url=cached_assets_base_url,
            assets_directory=cached_assets_directory,
            overwrite=False,
            s3_client=s3_client,
            s3_bucket=cached_assets_s3_bucket,
            s3_folder_name=cached_assets_s3_folder_name,
        )
        if use_s3_storage
        else DirectoryStorageOptions(
            assets_base_url=cached_assets_base_url, assets_directory=cached_assets_directory, overwrite=True
        )
    )
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(
            pa_table=pa_table,
            dataset=dataset,
            config=config,
            split=split,
            storage_options=storage_options,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
        ),
        "num_rows_total": num_rows_total,
        "num_rows_per_page": MAX_NUM_ROWS_PER_PAGE,
    }
