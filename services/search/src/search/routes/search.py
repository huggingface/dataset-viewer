# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
import random
from http import HTTPStatus
from typing import Optional

import anyio
import pyarrow as pa
from datasets import Features, Value
from libapi.authentication import auth_check
from libapi.duckdb import (
    get_cache_entry_from_parquet_metadata_job,
    get_index_file_location_and_build_if_missing,
)
from libapi.exceptions import (
    ApiError,
    UnexpectedApiError,
)
from libapi.request import (
    get_request_parameter,
    get_request_parameter_length,
    get_request_parameter_offset,
)
from libapi.utils import (
    Endpoint,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    to_rows_list,
)
from libcommon.constants import HF_FTS_SCORE, MAX_NUM_ROWS_PER_PAGE, ROW_IDX_COLUMN
from libcommon.dtos import PaginatedResponse
from libcommon.prometheus import StepProfiler
from libcommon.storage import StrPath, clean_dir
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.features import (
    get_supported_unsupported_columns,
    to_features_list,
)
from starlette.requests import Request
from starlette.responses import Response

from search.duckdb_connection import duckdb_connect_readonly

logger = logging.getLogger(__name__)

FTS_STAGE_TABLE_COMMAND = f"SELECT * FROM (SELECT {ROW_IDX_COLUMN}, fts_main_data.match_bm25({ROW_IDX_COLUMN}, ?) AS {HF_FTS_SCORE} FROM data) A WHERE {HF_FTS_SCORE} IS NOT NULL;"  # nosec
JOIN_STAGE_AND_DATA_COMMAND = "SELECT {columns} FROM memory.fts_stage_table JOIN db.data USING({row_idx_column}) ORDER BY memory.fts_stage_table.{hf_fts_score} DESC;"  # nosec
# ^ "no sec" to remove https://bandit.readthedocs.io/en/1.7.5/plugins/b608_hardcoded_sql_expressions.html
# the string substitutions here are constants, not user inputs


def full_text_search(
    index_file_location: str,
    columns: list[str],
    query: str,
    offset: int,
    length: int,
    extensions_directory: Optional[str] = None,
) -> tuple[int, pa.Table]:
    with duckdb_connect_readonly(extensions_directory=extensions_directory, database=index_file_location) as con:
        fts_stage_table = con.execute(query=FTS_STAGE_TABLE_COMMAND, parameters=[query]).arrow()
        num_rows_total = fts_stage_table.num_rows
        logging.info(f"got {num_rows_total=} results for {query=} using {offset=} {length=}")
        fts_stage_table = fts_stage_table.sort_by([(HF_FTS_SCORE, "descending")]).slice(offset, length)
        join_stage_and_data_query = JOIN_STAGE_AND_DATA_COMMAND.format(
            columns=",".join([f'"{column}"' for column in columns]),
            row_idx_column=ROW_IDX_COLUMN,
            hf_fts_score=HF_FTS_SCORE,
        )
        con.execute("USE memory;")
        con.from_arrow(fts_stage_table).create_view("fts_stage_table")
        con.execute("USE db;")
        pa_table = con.execute(query=join_stage_and_data_query).arrow()
    return num_rows_total, pa_table


async def create_response(
    pa_table: pa.Table,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    storage_client: StorageClient,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    num_rows_total: int,
    partial: bool,
) -> PaginatedResponse:
    features_without_key = features.copy()
    features_without_key.pop(ROW_IDX_COLUMN, None)
    if len(pa_table) > 0:
        pa_table = pa_table.drop(unsupported_columns)
    logging.info(f"create response for {dataset=} {config=} {split=}")

    return PaginatedResponse(
        features=to_features_list(features_without_key),
        rows=await to_rows_list(
            pa_table=pa_table,
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            storage_client=storage_client,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
            row_idx_column=ROW_IDX_COLUMN,
        ),
        num_rows_total=num_rows_total,
        num_rows_per_page=MAX_NUM_ROWS_PER_PAGE,
        partial=partial,
    )


def create_search_endpoint(
    duckdb_index_file_directory: StrPath,
    cached_assets_storage_client: StorageClient,
    parquet_metadata_directory: StrPath,
    hf_endpoint: str,
    blocked_datasets: list[str],
    external_auth_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    storage_clients: Optional[list[StorageClient]] = None,
    extensions_directory: Optional[str] = None,
    clean_cache_proba: float = 0.0,
    expiredTimeIntervalSeconds: int = 60,
    max_split_size_bytes: int = 5_000_000_000,
) -> Endpoint:
    async def search_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="search_endpoint", step="all"):
            try:
                with StepProfiler(method="search_endpoint", step="validate parameters"):
                    dataset = get_request_parameter(request, "dataset", required=True)
                    config = get_request_parameter(request, "config", required=True)
                    split = get_request_parameter(request, "split", required=True)
                    query = get_request_parameter(request, "query", required=True)
                    offset = get_request_parameter_offset(request)
                    length = get_request_parameter_length(request)

                with StepProfiler(method="search_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    await auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )

                logging.info(f"/search {dataset=} {config=} {split=} {query=} {offset=} {length=}")

                with StepProfiler(method="filter_endpoint", step="build index if missing"):
                    # get parquet urls and dataset_info
                    parquet_metadata_response = get_cache_entry_from_parquet_metadata_job(
                        dataset=dataset,
                        config=config,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        hf_timeout_seconds=hf_timeout_seconds,
                        blocked_datasets=blocked_datasets,
                        storage_clients=storage_clients,
                    )
                    revision = parquet_metadata_response["dataset_git_revision"]
                    if parquet_metadata_response["http_status"] != HTTPStatus.OK:
                        return get_json_error_response(
                            content=parquet_metadata_response["content"],
                            status_code=parquet_metadata_response["http_status"],
                            max_age=max_age_short,
                            error_code=parquet_metadata_response["error_code"],
                            revision=revision,
                        )
                    content_parquet_metadata = parquet_metadata_response["content"]
                    split_parquet_files = [
                        parquet_file
                        for parquet_file in content_parquet_metadata["parquet_files_metadata"]
                        if parquet_file["config"] == config and parquet_file["split"] == split
                    ]
                    index_file_location, partial = await get_index_file_location_and_build_if_missing(
                        duckdb_index_file_directory=duckdb_index_file_directory,
                        dataset=dataset,
                        config=config,
                        split=split,
                        revision=revision,
                        hf_token=hf_token,
                        max_split_size_bytes=max_split_size_bytes,
                        extensions_directory=extensions_directory,
                        parquet_metadata_directory=parquet_metadata_directory,
                        split_parquet_files=split_parquet_files,
                        features=content_parquet_metadata["features"],
                    )
                    # features must contain the row idx column for full_text_search
                    features = Features.from_dict(content_parquet_metadata["features"])
                    features[ROW_IDX_COLUMN] = Value("int64")
                with StepProfiler(method="search_endpoint", step="get supported and unsupported columns"):
                    supported_columns, unsupported_columns = get_supported_unsupported_columns(
                        features,
                    )
                with StepProfiler(method="search_endpoint", step="perform FTS command"):
                    logging.debug(f"connect to index file {index_file_location}")
                    num_rows_total, pa_table = await anyio.to_thread.run_sync(
                        full_text_search,
                        index_file_location,
                        supported_columns,
                        query,
                        offset,
                        length,
                        extensions_directory,
                    )
                    # no need to do it every time
                    # TODO: Will be moved to another process in parallel
                    if random.random() < clean_cache_proba:  # nosec
                        with StepProfiler(method="search_endpoint", step="clean old indexes"):
                            clean_dir(
                                duckdb_index_file_directory,
                                expiredTimeIntervalSeconds,
                            )
                with StepProfiler(method="search_endpoint", step="create response"):
                    response = await create_response(
                        pa_table=pa_table,
                        dataset=dataset,
                        revision=revision,
                        config=config,
                        split=split,
                        storage_client=cached_assets_storage_client,
                        offset=offset,
                        features=features or Features.from_arrow_schema(pa_table.schema),
                        unsupported_columns=unsupported_columns,
                        num_rows_total=num_rows_total,
                        partial=partial,
                    )
                    logging.info(f"transform rows finished for {dataset=} {config=} {split=}")
                with StepProfiler(method="search_endpoint", step="generate the OK response"):
                    return get_json_ok_response(response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="search_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return search_endpoint
