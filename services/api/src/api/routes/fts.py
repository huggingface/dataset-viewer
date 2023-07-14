# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import json
import logging
import re
from hashlib import sha1
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import duckdb
from datasets import Features
from huggingface_hub import hf_hub_download
from libapi.exceptions import (
    ApiError,
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedApiError,
)
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    to_rows_list,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.storage import StrPath, init_dir
from libcommon.viewer_utils.features import to_features_list
from starlette.requests import Request
from starlette.responses import Response

from api.routes.endpoint import get_cache_entry_from_steps

DUCKDB_DEFAULT_INDEX_FILENAME = "index.duckdb"
MAX_ROWS = 100
FTS_COMMAND = (
    "SELECT * EXCLUDE (__hf_index_id, score) FROM (SELECT *, fts_main_data.match_bm25(__hf_index_id, ?) AS score FROM"
    " data) A WHERE score IS NOT NULL ORDER BY __hf_index_id OFFSET {offset} LIMIT {length};"
)


def create_fts_endpoint(
    processing_graph: ProcessingGraph,
    duckdb_index_file_directory: StrPath,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    target_revision: str,
    cache_max_days: int,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def fts_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="fts_endpoint", step="all"):
            try:
                dataset = request.query_params.get("dataset")
                config = request.query_params.get("config")
                split = request.query_params.get("split")
                query = request.query_params.get("query")

                # TODO: Evaluate query parameter (Prevent SQL injection)
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

                length = int(request.query_params.get("length", MAX_ROWS))
                if length < 0:
                    raise InvalidParameterError("Length must be positive")
                if length > MAX_ROWS:
                    raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")

                logging.info(f"/fts {dataset=} {config=} {split=} {query=} {offset=} {length=}")

                # TODO: Symplify temporal index folder name
                payload = ("download", dataset, config, split)
                hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
                prefix = f"download-{dataset}"[:64]
                subdirectory = f"{prefix}-{hash_suffix}"
                index_folder = "".join([c if re.match(r"[\w-]", c) else "-" for c in subdirectory])
                index_folder = f"{duckdb_index_file_directory}/{index_folder}"
                filename = f"{config}/{split}/{DUCKDB_DEFAULT_INDEX_FILENAME}"
                index_file_location = f"{index_folder}/{filename}"
                index_exists = Path(index_file_location).is_file()

                # TODO: Note that actually, no cache data is needed to download the index file
                processing_steps = processing_graph.get_processing_step_by_job_type("config-info")
                result = get_cache_entry_from_steps(
                    processing_steps=[processing_steps],
                    dataset=dataset,
                    config=config,
                    split=None,
                    processing_graph=processing_graph,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    hf_timeout_seconds=hf_timeout_seconds,
                    cache_max_days=cache_max_days,
                )

                content = result["content"]
                features = content["dataset_info"]["features"]
                http_status = result["http_status"]
                error_code = result["error_code"]
                revision = result["dataset_git_revision"]
                if http_status != HTTPStatus.OK:
                    return get_json_error_response(
                        content=content,
                        status_code=http_status,
                        max_age=max_age_short,
                        error_code=error_code,
                        revision=revision,
                    )

                if not index_exists:
                    logging.info(f"init_dir {index_folder}")
                    init_dir(index_folder)
                    hf_hub_download(
                        repo_type="dataset",
                        revision=target_revision,
                        repo_id=dataset,
                        filename=filename,
                        local_dir=index_folder,
                        local_dir_use_symlinks=False,
                        token=hf_token,
                    )
                con = duckdb.connect(index_file_location, read_only=True)
                query_result = con.execute(
                    (FTS_COMMAND.format(offset=offset, length=length)),
                    [query],
                )  # TODO: Does the response need to have the totalCount for pagination?
                pa_table = query_result.arrow()
                con.close()

                features = Features.from_arrow_schema(pa_table.schema)
                response = {
                    "features": to_features_list(features),
                    "rows": to_rows_list(
                        pa_table,
                        dataset,
                        config,
                        split,
                        cached_assets_base_url,
                        cached_assets_directory,
                        offset=offset,
                        features=features,
                        unsupported_columns=[],
                    ),
                }
                return get_json_ok_response(response, max_age=max_age_long)
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="processing_step_endpoint", step="generate API error response", context=""):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return fts_endpoint
