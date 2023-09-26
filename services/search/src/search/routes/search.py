# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
from datasets import Features
from libapi.authentication import auth_check
from libapi.exceptions import (
    ApiError,
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedApiError,
)
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    clean_cached_assets_randomly,
    get_cache_entry_from_steps,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    to_rows_list,
)
from libcommon.duckdb import get_index_file_location_and_download_if_missing
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CacheEntry
from libcommon.storage import StrPath
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE, PaginatedResponse
from libcommon.viewer_utils.features import (
    get_supported_unsupported_columns,
    to_features_list,
)
from starlette.requests import Request
from starlette.responses import Response

ROW_IDX_COLUMN = "__hf_index_id"

FTS_COMMAND_COUNT = (
    "SELECT COUNT(*) FROM (SELECT __hf_index_id, fts_main_data.match_bm25(__hf_index_id, ?) AS __hf_fts_score FROM"
    " data) A WHERE __hf_fts_score IS NOT NULL;"
)

FTS_COMMAND = (
    "SELECT * EXCLUDE (__hf_fts_score) FROM (SELECT *, fts_main_data.match_bm25(__hf_index_id, ?) AS __hf_fts_score"
    " FROM data) A WHERE __hf_fts_score IS NOT NULL ORDER BY __hf_fts_score DESC OFFSET {offset} LIMIT {length};"
)
REPO_TYPE = "dataset"
HUB_DOWNLOAD_CACHE_FOLDER = "cache"


logger = logging.getLogger(__name__)


def full_text_search(index_file_location: str, query: str, offset: int, length: int) -> tuple[int, pa.Table]:
    con = duckdb.connect(index_file_location, read_only=True)
    count_result = con.execute(query=FTS_COMMAND_COUNT, parameters=[query]).fetchall()
    num_rows_total = count_result[0][0]  # it will always return a non-empty list with one element in a tuple
    logging.debug(f"got {num_rows_total=} results for {query=}")
    query_result = con.execute(
        query=FTS_COMMAND.format(offset=offset, length=length),
        parameters=[query],
    )
    pa_table = query_result.arrow()
    con.close()
    return num_rows_total, pa_table


def create_response(
    pa_table: pa.Table,
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
    features: Features,
    num_rows_total: int,
) -> PaginatedResponse:
    features_without_key = features.copy()
    features_without_key.pop(ROW_IDX_COLUMN, None)

    _, unsupported_columns = get_supported_unsupported_columns(
        features,
    )
    pa_table = pa_table.drop(unsupported_columns)
    return PaginatedResponse(
        features=to_features_list(features_without_key),
        rows=to_rows_list(
            pa_table,
            dataset,
            config,
            split,
            cached_assets_base_url,
            cached_assets_directory,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
            row_idx_column=ROW_IDX_COLUMN,
        ),
        num_rows_total=num_rows_total,
        num_rows_per_page=MAX_NUM_ROWS_PER_PAGE,
    )


def create_search_endpoint(
    processing_graph: ProcessingGraph,
    duckdb_index_file_directory: StrPath,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    target_revision: str,
    cache_max_days: int,
    hf_endpoint: str,
    external_auth_url: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    clean_cache_proba: float = 0.0,
    keep_first_rows_number: int = -1,
    keep_most_recent_rows_number: int = -1,
    max_cleaned_rows_number: int = -1,
) -> Endpoint:
    async def search_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="search_endpoint", step="all"):
            try:
                with StepProfiler(method="search_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    query = request.query_params.get("query")

                    if (
                        not dataset
                        or not config
                        or not split
                        or not query
                        or not are_valid_parameters([dataset, config, split, query])
                    ):
                        raise MissingRequiredParameterError(
                            "Parameter 'dataset', 'config', 'split' and 'query' are required"
                        )

                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Offset must be positive")

                    length = int(request.query_params.get("length", MAX_NUM_ROWS_PER_PAGE))
                    if length < 0:
                        raise InvalidParameterError("Length must be positive")
                    if length > MAX_NUM_ROWS_PER_PAGE:
                        raise InvalidParameterError(
                            f"Parameter 'length' must not be bigger than {MAX_NUM_ROWS_PER_PAGE}"
                        )

                with StepProfiler(method="search_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )

                logging.info(f"/search {dataset=} {config=} {split=} {query=} {offset=} {length=}")

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

                with StepProfiler(method="search_endpoint", step="perform FTS command"):
                    logging.debug(f"connect to index file {index_file_location}")
                    (num_rows_total, pa_table) = full_text_search(index_file_location, query, offset, length)
                    Path(index_file_location).touch()

                with StepProfiler(method="search_endpoint", step="clean cache randomly"):
                    clean_cached_assets_randomly(
                        clean_cache_proba=clean_cache_proba,
                        dataset=dataset,
                        cached_assets_directory=cached_assets_directory,
                        keep_first_rows_number=keep_first_rows_number,
                        keep_most_recent_rows_number=keep_most_recent_rows_number,
                        max_cleaned_rows_number=max_cleaned_rows_number,
                    )

                with StepProfiler(method="search_endpoint", step="create response"):
                    if "features" in duckdb_index_cache_entry["content"] and isinstance(
                        duckdb_index_cache_entry["content"]["features"], dict
                    ):
                        features = Features.from_dict(duckdb_index_cache_entry["content"]["features"])
                    else:
                        features = Features.from_arrow_schema(pa_table.schema)
                    response = create_response(
                        pa_table,
                        dataset,
                        config,
                        split,
                        cached_assets_base_url,
                        cached_assets_directory,
                        offset,
                        features,
                        num_rows_total,
                    )
                with StepProfiler(method="search_endpoint", step="generate the OK response"):
                    return get_json_ok_response(response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="search_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return search_endpoint


def get_cache_entry_from_duckdb_index_job(
    processing_graph: ProcessingGraph,
    dataset: str,
    config: str,
    split: str,
    hf_endpoint: str,
    hf_token: Optional[str],
    hf_timeout_seconds: Optional[float],
    cache_max_days: int,
) -> CacheEntry:
    processing_steps = processing_graph.get_processing_step_by_job_type("split-duckdb-index")
    return get_cache_entry_from_steps(
        processing_steps=[processing_steps],
        dataset=dataset,
        config=config,
        split=split,
        processing_graph=processing_graph,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        cache_max_days=cache_max_days,
    )
