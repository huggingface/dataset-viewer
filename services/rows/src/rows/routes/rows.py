# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Literal, Optional, Union

from fsspec.implementations.http import HTTPFileSystem
from libapi.authentication import auth_check
from libapi.exceptions import ApiError, UnexpectedApiError
from libapi.request import (
    get_request_parameter_length,
    get_request_parameter_offset,
    get_required_request_parameter,
)
from libapi.response import create_response
from libapi.utils import (
    Endpoint,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
    try_backfill_dataset_then_raise,
)
from libcommon.parquet_utils import Indexer
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.s3_client import S3Client
from libcommon.simple_cache import CachedArtifactError, CachedArtifactNotFoundError
from libcommon.storage import StrPath
from libcommon.viewer_utils.features import UNSUPPORTED_FEATURES
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


ALL_COLUMNS_SUPPORTED_DATASETS_ALLOW_LIST: Union[Literal["all"], list[str]] = ["arabic_speech_corpus"]  # for testing


def create_rows_endpoint(
    processing_graph: ProcessingGraph,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    s3_client: S3Client,
    cached_assets_s3_folder_name: str,
    parquet_metadata_directory: StrPath,
    cache_max_days: int,
    max_arrow_data_in_memory: int,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    indexer = Indexer(
        processing_graph=processing_graph,
        hf_token=hf_token,
        parquet_metadata_directory=parquet_metadata_directory,
        httpfs=HTTPFileSystem(headers={"authorization": f"Bearer {hf_token}"}),
        max_arrow_data_in_memory=max_arrow_data_in_memory,
        unsupported_features=UNSUPPORTED_FEATURES,
        all_columns_supported_datasets_allow_list=ALL_COLUMNS_SUPPORTED_DATASETS_ALLOW_LIST,
    )

    async def rows_endpoint(request: Request) -> Response:
        await indexer.httpfs.set_session()
        revision: Optional[str] = None
        with StepProfiler(method="rows_endpoint", step="all"):
            try:
                with StepProfiler(method="rows_endpoint", step="validate parameters"):
                    dataset = get_required_request_parameter(request, "dataset")
                    config = get_required_request_parameter(request, "config")
                    split = get_required_request_parameter(request, "split")
                    offset = get_request_parameter_offset(request)
                    length = get_request_parameter_length(request)
                    logging.info(
                        f"/rows, dataset={dataset}, config={config}, split={split}, offset={offset}, length={length}"
                    )
                with StepProfiler(method="rows_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    await auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                try:
                    with StepProfiler(method="rows_endpoint", step="get row groups index"):
                        rows_index = indexer.get_rows_index(
                            dataset=dataset,
                            config=config,
                            split=split,
                        )
                        revision = rows_index.revision
                    with StepProfiler(method="rows_endpoint", step="query the rows"):
                        pa_table = rows_index.query(offset=offset, length=length)
                    with StepProfiler(method="rows_endpoint", step="transform to a list"):
                        response = create_response(
                            dataset=dataset,
                            revision=revision,
                            config=config,
                            split=split,
                            cached_assets_base_url=cached_assets_base_url,
                            cached_assets_directory=cached_assets_directory,
                            s3_client=s3_client,
                            cached_assets_s3_folder_name=cached_assets_s3_folder_name,
                            pa_table=pa_table,
                            offset=offset,
                            features=rows_index.parquet_index.features,
                            unsupported_columns=rows_index.parquet_index.unsupported_columns,
                            num_rows_total=rows_index.parquet_index.num_rows_total,
                        )
                except CachedArtifactNotFoundError:
                    config_parquet_processing_steps = processing_graph.get_config_parquet_processing_steps()
                    config_parquet_metadata_processing_steps = (
                        processing_graph.get_config_parquet_metadata_processing_steps()
                    )
                    with StepProfiler(method="rows_endpoint", step="try backfill dataset"):
                        try_backfill_dataset_then_raise(
                            processing_steps=config_parquet_metadata_processing_steps
                            + config_parquet_processing_steps,
                            processing_graph=processing_graph,
                            dataset=dataset,
                            hf_endpoint=hf_endpoint,
                            hf_timeout_seconds=hf_timeout_seconds,
                            hf_token=hf_token,
                            cache_max_days=cache_max_days,
                            blocked_datasets=blocked_datasets,
                        )
                with StepProfiler(method="rows_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except CachedArtifactError as e:
                content = e.cache_entry_with_details["content"]
                http_status = e.cache_entry_with_details["http_status"]
                error_code = e.cache_entry_with_details["error_code"]
                return get_json_error_response(
                    content=content,
                    status_code=http_status,
                    max_age=max_age_short,
                    error_code=error_code,
                    revision=revision,
                )
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return rows_endpoint
