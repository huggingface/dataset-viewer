# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

import duckdb
import pyarrow as pa
from datasets import Features
from libapi.authentication import auth_check
from libapi.duckdb import (
    get_cache_entry_from_duckdb_index_job,
    get_index_file_location_and_download_if_missing,
)
from libapi.exceptions import (
    ApiError,
    InvalidParameterError,
    MissingRequiredParameterError,
)
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    clean_cached_assets_randomly,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    to_rows_list,
)
from libcommon.exceptions import UnexpectedError
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.s3_client import S3Client
from libcommon.storage import StrPath
from libcommon.storage_options import DirectoryStorageOptions, S3StorageOptions
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE, PaginatedResponse
from libcommon.viewer_utils.features import (
    get_supported_unsupported_columns,
    to_features_list,
)
from starlette.requests import Request
from starlette.responses import Response

FILTER_QUERY = """\
    SELECT {columns}
    FROM data
    WHERE {where}
    LIMIT {limit}
    OFFSET {offset}"""

FILTER_COUNT_QUERY = """\
    SELECT COUNT(*)
    FROM data
    WHERE {where}"""

CACHED_ASSETS_S3_SUPPORTED_DATASETS: list[str] = ["asoria/image"]  # for testing

logger = logging.getLogger(__name__)


def create_filter_endpoint(
    processing_graph: ProcessingGraph,
    duckdb_index_file_directory: StrPath,
    target_revision: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    s3_client: S3Client,
    cached_assets_s3_bucket: str,
    cached_assets_s3_folder_name: str,
    cache_max_days: int,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    clean_cache_proba: float = 0.0,
    keep_first_rows_number: int = -1,
    keep_most_recent_rows_number: int = -1,
    max_cleaned_rows_number: int = -1,
) -> Endpoint:
    async def filter_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="filter_endpoint", step="all"):
            try:
                with StepProfiler(method="filter_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    where = request.query_params.get("where")
                    if (
                        not dataset
                        or not config
                        or not split
                        or not where
                        or not are_valid_parameters([dataset, config, split, where])
                    ):
                        raise MissingRequiredParameterError(
                            "Parameters 'dataset', 'config', 'split' and 'where' are required"
                        )
                    # TODO: validate where
                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Parameter 'offset' must be positive")
                    length = int(request.query_params.get("length", MAX_NUM_ROWS_PER_PAGE))
                    if length < 0:
                        raise InvalidParameterError("Parameter 'length' must be positive")
                    elif length > MAX_NUM_ROWS_PER_PAGE:
                        raise InvalidParameterError(
                            f"Parameter 'length' must not be bigger than {MAX_NUM_ROWS_PER_PAGE}"
                        )
                    logger.info(
                        f'/filter, dataset={dataset}, config={config}, split={split}, where="{where}",'
                        f" offset={offset}, length={length}"
                    )
                with StepProfiler(method="filter_endpoint", step="check authentication"):
                    # If auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                # TODO: duplicated in /search
                with StepProfiler(method="search_endpoint", step="validate indexing was done"):
                    # no cache data is needed to download the index file
                    # but will help to validate if indexing was done
                    duckdb_index_cache_entry = get_cache_entry_from_duckdb_index_job(
                        processing_graph=processing_graph,
                        dataset=dataset,
                        config=config,
                        split=split,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        hf_timeout_seconds=hf_timeout_seconds,
                        cache_max_days=cache_max_days,
                    )
                    revision = duckdb_index_cache_entry["dataset_git_revision"]
                    if duckdb_index_cache_entry["http_status"] != HTTPStatus.OK:
                        return get_json_error_response(
                            content=duckdb_index_cache_entry["content"],
                            status_code=duckdb_index_cache_entry["http_status"],
                            max_age=max_age_short,
                            error_code=duckdb_index_cache_entry["error_code"],
                            revision=revision,
                        )
                with StepProfiler(method="search_endpoint", step="download index file if missing"):
                    index_file_location = get_index_file_location_and_download_if_missing(
                        duckdb_index_file_directory=duckdb_index_file_directory,
                        dataset=dataset,
                        config=config,
                        split=split,
                        revision=revision,
                        filename=duckdb_index_cache_entry["content"]["filename"],
                        url=duckdb_index_cache_entry["content"]["url"],
                        target_revision=target_revision,
                        hf_token=hf_token,
                    )
                with StepProfiler(method="filter_endpoint", step="get features"):
                    try:
                        features = Features.from_dict(
                            {
                                name: feature
                                for name, feature in duckdb_index_cache_entry["content"]["features"].items()
                                if name != "__hf_index_id"
                            }
                        )
                    except (KeyError, AttributeError):
                        raise RuntimeError("The indexing process did not store the features.")
                with StepProfiler(method="filter_endpoint", step="get supported and unsupported columns"):
                    supported_columns, unsupported_columns = get_supported_unsupported_columns(
                        features,
                    )
                with StepProfiler(method="filter_endpoint", step="execute filter query"):
                    num_rows_total, pa_table = execute_filter_query(
                        index_file_location=index_file_location,
                        columns=supported_columns,
                        where=where,
                        limit=length,
                        offset=offset,
                    )
                with StepProfiler(method="search_endpoint", step="clean cache randomly"):
                    clean_cached_assets_randomly(
                        clean_cache_proba=clean_cache_proba,
                        dataset=dataset,
                        cached_assets_directory=cached_assets_directory,
                        keep_first_rows_number=keep_first_rows_number,
                        keep_most_recent_rows_number=keep_most_recent_rows_number,
                        max_cleaned_rows_number=max_cleaned_rows_number,
                    )
                with StepProfiler(method="filter_endpoint", step="create response"):
                    response = create_response(
                        dataset=dataset,
                        config=config,
                        split=split,
                        cached_assets_base_url=cached_assets_base_url,
                        cached_assets_directory=cached_assets_directory,
                        s3_client=s3_client,
                        cached_assets_s3_bucket=cached_assets_s3_bucket,
                        cached_assets_s3_folder_name=cached_assets_s3_folder_name,
                        pa_table=pa_table,
                        offset=offset,
                        features=features,
                        unsupported_columns=unsupported_columns,
                        num_rows_total=num_rows_total,
                    )
                with StepProfiler(method="filter_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="filter_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return filter_endpoint


def execute_filter_query(
    index_file_location: str, columns: list[str], where: str, limit: int, offset: int
) -> tuple[int, pa.Table]:
    con = duckdb.connect(database=index_file_location, read_only=True)
    # TODO: Address possible SQL injection CWE-89
    filter_query = FILTER_QUERY.format(columns=",".join(columns), where=where, limit=limit, offset=offset)
    pa_table = con.sql(filter_query).arrow()
    filter_count_query = FILTER_COUNT_QUERY.format(where=where)
    num_rows_total = con.sql(filter_count_query).fetchall()[0][0]
    return num_rows_total, pa_table


# TODO: duplicated in /rows
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
            pa_table,
            dataset,
            config,
            split,
            storage_options=storage_options,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
        ),
        "num_rows_total": num_rows_total,
        "num_rows_per_page": MAX_NUM_ROWS_PER_PAGE,
    }
