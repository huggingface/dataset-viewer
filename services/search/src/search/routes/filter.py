# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import json
import logging
import os
import random
import re
from hashlib import sha1
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
from datasets import Features
from huggingface_hub import hf_hub_download
from libapi.authentication import auth_check
from libapi.exceptions import (
    ApiError,
    InvalidParameterError,
    MissingRequiredParameterError,
)
from libapi.utils import (
    Endpoint,
    are_valid_parameters,
    clean_cached_assets,
    get_cache_entry_from_steps,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    to_rows_list,
)
from libcommon.exceptions import UnexpectedError
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.storage import StrPath, init_dir
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE, PaginatedResponse
from libcommon.viewer_utils.features import (
    get_supported_unsupported_columns,
    to_features_list,
)
from starlette.requests import Request
from starlette.responses import Response

REPO_TYPE = "dataset"
HUB_DOWNLOAD_CACHE_FOLDER = "cache"

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

logger = logging.getLogger(__name__)


def create_filter_endpoint(
    processing_graph: ProcessingGraph,
    duckdb_index_file_directory: StrPath,
    target_revision: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
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
                    processing_steps = processing_graph.get_processing_step_by_job_type("split-duckdb-index")
                    result = get_cache_entry_from_steps(
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
                    content = result["content"]
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
                # TODO: duplicated in /search
                with StepProfiler(method="search_endpoint", step="download index file if missing"):
                    file_name = content["filename"]
                    index_folder = get_download_folder(duckdb_index_file_directory, dataset, config, split, revision)
                    # For directories like "partial-train" for the file
                    # at "en/partial-train/0000.parquet" in the C4 dataset.
                    # Note that "-" is forbidden for split names, so it doesn't create directory names collisions.
                    split_directory = content["url"].rsplit("/", 2)[1]
                    repo_file_location = f"{config}/{split_directory}/{file_name}"
                    index_file_location = f"{index_folder}/{repo_file_location}"
                    index_path = Path(index_file_location)
                    if not index_path.is_file():
                        with StepProfiler(method="search_endpoint", step="download index file"):
                            download_index_file(
                                cache_folder=f"{duckdb_index_file_directory}/{HUB_DOWNLOAD_CACHE_FOLDER}",
                                index_folder=index_folder,
                                target_revision=target_revision,
                                dataset=dataset,
                                repo_file_location=repo_file_location,
                                hf_token=hf_token,
                            )
                with StepProfiler(method="filter_endpoint", step="get features"):
                    try:
                        features = Features.from_dict(
                            {name: feature for name, feature in content["features"].items() if name != "__hf_index_id"}
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


# TODO: duplicated in /search
def get_download_folder(
    root_directory: StrPath, dataset: str, config: str, split: str, revision: Optional[str]
) -> str:
    payload = (dataset, config, split, revision)
    hash_suffix = sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]
    subdirectory = "".join([c if re.match(r"[\w-]", c) else "-" for c in f"{dataset}-{hash_suffix}"])
    return f"{root_directory}/downloads/{subdirectory}"


# TODO: duplicated in /search
def download_index_file(
    cache_folder: str,
    index_folder: str,
    target_revision: str,
    dataset: str,
    repo_file_location: str,
    hf_token: Optional[str] = None,
) -> None:
    logging.info(f"init_dir {index_folder}")
    init_dir(index_folder)

    # see https://pypi.org/project/hf-transfer/ for more details about how to enable hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    hf_hub_download(
        repo_type=REPO_TYPE,
        revision=target_revision,
        repo_id=dataset,
        filename=repo_file_location,
        local_dir=index_folder,
        local_dir_use_symlinks=False,
        token=hf_token,
        cache_dir=cache_folder,
    )


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


# TODO: duplicated in /search
def clean_cached_assets_randomly(
    clean_cache_proba: float,
    dataset: str,
    cached_assets_directory: StrPath,
    keep_first_rows_number: int,
    keep_most_recent_rows_number: int,
    max_cleaned_rows_number: int,
) -> None:
    # no need to do it every time
    if random.random() < clean_cache_proba:  # nosec
        if (
                keep_first_rows_number < 0
                and keep_most_recent_rows_number < 0
                and max_cleaned_rows_number < 0
        ):
            logger.debug(
                "Params keep_first_rows_number, keep_most_recent_rows_number and"
                " max_cleaned_rows_number are not set. Skipping cached assets cleaning."
            )
        else:
            clean_cached_assets(
                dataset=dataset,
                cached_assets_directory=cached_assets_directory,
                keep_first_rows_number=keep_first_rows_number,
                keep_most_recent_rows_number=keep_most_recent_rows_number,
                max_cleaned_rows_number=max_cleaned_rows_number,
            )


# TODO: duplicated in /rows
def create_response(
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    num_rows_total: int,
) -> PaginatedResponse:
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(
            pa_table,
            dataset,
            config,
            split,
            cached_assets_base_url,
            cached_assets_directory,
            offset,
            features,
            unsupported_columns,
        ),
        "num_rows_total": num_rows_total,
        "num_rows_per_page": MAX_NUM_ROWS_PER_PAGE,
    }
