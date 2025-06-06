# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
import random
import re
from http import HTTPStatus
from typing import Optional

import anyio
import duckdb
import pyarrow as pa
from datasets import Features, Value
from libapi.authentication import auth_check
from libapi.duckdb import (
    get_cache_entry_from_parquet_metadata_job,
    get_index_file_location_and_build_if_missing,
)
from libapi.exceptions import ApiError, InvalidParameterError, UnexpectedApiError
from libapi.request import (
    get_request_parameter,
    get_request_parameter_length,
    get_request_parameter_offset,
)
from libapi.response import create_response
from libapi.utils import (
    Endpoint,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)
from libcommon.constants import ROW_IDX_COLUMN
from libcommon.prometheus import StepProfiler
from libcommon.storage import StrPath, clean_dir
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.features import get_supported_unsupported_columns
from starlette.requests import Request
from starlette.responses import Response

from search.duckdb_connection import duckdb_connect_readonly

FILTER_QUERY = """\
    SELECT {columns}
    FROM data
    {where}
    {orderby}
    LIMIT {limit}
    OFFSET {offset}"""

FILTER_COUNT_QUERY = """\
    SELECT COUNT(*)
    FROM data
    {where}"""

SQL_INVALID_SYMBOLS = "|".join([";", "--", r"/\*", r"\*/"])
SQL_INVALID_SYMBOLS_PATTERN = re.compile(rf"(?:{SQL_INVALID_SYMBOLS})", flags=re.IGNORECASE)

logger = logging.getLogger(__name__)


def create_filter_endpoint(
    duckdb_index_file_directory: StrPath,
    cached_assets_storage_client: StorageClient,
    parquet_metadata_directory: StrPath,
    blocked_datasets: list[str],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    storage_clients: Optional[list[StorageClient]] = None,
    extensions_directory: Optional[str] = None,
    clean_cache_proba: float = 0.0,
    expiredTimeIntervalSeconds: int = 60,
    max_split_size_bytes: int = 5_000_000_000,
) -> Endpoint:
    async def filter_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="filter_endpoint", step="all"):
            try:
                with StepProfiler(method="filter_endpoint", step="validate parameters"):
                    dataset = get_request_parameter(request, "dataset", required=True)
                    config = get_request_parameter(request, "config", required=True)
                    split = get_request_parameter(request, "split", required=True)
                    where = get_request_parameter(request, "where")
                    validate_query_parameter(where, "where")
                    orderby = get_request_parameter(request, "orderby")
                    validate_query_parameter(orderby, "orderby")
                    offset = get_request_parameter_offset(request)
                    length = get_request_parameter_length(request)
                    logger.info(
                        f"/filter, {dataset=}, {config=}, {split=}, {where=}, {orderby=}, {offset=}, {length=}"
                    )
                with StepProfiler(method="filter_endpoint", step="check authentication"):
                    # If auth_check fails, it will raise an exception that will be caught below
                    await auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )

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
                with StepProfiler(method="filter_endpoint", step="get supported and unsupported columns"):
                    supported_columns, unsupported_columns = get_supported_unsupported_columns(
                        features,
                    )
                with StepProfiler(method="filter_endpoint", step="execute filter query"):
                    num_rows_total, pa_table = await anyio.to_thread.run_sync(
                        execute_filter_query,
                        index_file_location,
                        supported_columns,
                        where,
                        orderby,
                        length,
                        offset,
                        extensions_directory,
                    )
                    # no need to do it every time
                    # TODO: Will be moved to another process in parallel
                    if random.random() < clean_cache_proba:  # nosec
                        with StepProfiler(method="filter_endpoint", step="clean old indexes"):
                            clean_dir(
                                duckdb_index_file_directory,
                                expiredTimeIntervalSeconds,
                            )
                with StepProfiler(method="filter_endpoint", step="create response"):
                    response = await create_response(
                        dataset=dataset,
                        revision=revision,
                        config=config,
                        split=split,
                        storage_client=cached_assets_storage_client,
                        pa_table=pa_table,
                        offset=offset,
                        features=features or Features.from_arrow_schema(pa_table.schema),
                        unsupported_columns=unsupported_columns,
                        num_rows_total=num_rows_total,
                        partial=partial,
                        use_row_idx_column=True,
                    )
                with StepProfiler(method="filter_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="filter_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return filter_endpoint


def execute_filter_query(
    index_file_location: str,
    columns: list[str],
    where: str,
    orderby: str,
    limit: int,
    offset: int,
    extensions_directory: Optional[str] = None,
) -> tuple[int, pa.Table]:
    with duckdb_connect_readonly(extensions_directory=extensions_directory, database=index_file_location) as con:
        filter_query = FILTER_QUERY.format(
            columns=",".join([f'"{column}"' for column in columns]),
            where=f"WHERE {where}" if where else "",
            orderby=f"ORDER BY {orderby}" if orderby else "",
            limit=limit,
            offset=offset,
        )
        filter_count_query = FILTER_COUNT_QUERY.format(where=f"WHERE {where}" if where else "")
        try:
            pa_table = con.sql(filter_query).arrow()
            num_rows_total = con.sql(filter_count_query).fetchall()[0][0]
        except duckdb.Error as err:
            raise InvalidParameterError(message="A query parameter is invalid") from err
    return num_rows_total, pa_table


def validate_query_parameter(parameter_value: str, parameter_name: str) -> None:
    if SQL_INVALID_SYMBOLS_PATTERN.search(parameter_value):
        raise InvalidParameterError(message=f"Parameter '{parameter_name}' contains invalid symbols")
