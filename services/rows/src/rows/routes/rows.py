# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
from typing import List, Literal, Optional, Union

import pyarrow as pa
from datasets import Audio, Features, Value
from fsspec.implementations.http import HTTPFileSystem
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
    clean_cached_assets,
    get_json_api_error_response,
    get_json_ok_response,
    to_rows_list,
    try_backfill_dataset_then_raise,
)
from libcommon.parquet_utils import Indexer
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CachedArtifactError
from libcommon.storage import StrPath
from libcommon.utils import PaginatedResponse
from libcommon.viewer_utils.asset import update_last_modified_date_of_rows_in_assets_dir
from libcommon.viewer_utils.features import to_features_list
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


MAX_ROWS = 100


ALL_COLUMNS_SUPPORTED_DATASETS_ALLOW_LIST: Union[Literal["all"], List[str]] = ["arabic_speech_corpus"]  # for testing

# audio still has some errors when librosa is imported
UNSUPPORTED_FEATURES = [Value("binary"), Audio()]


def create_response(
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: List[str],
    num_rows_total: int,
) -> PaginatedResponse:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    return PaginatedResponse(
        features=to_features_list(features),
        rows=to_rows_list(
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
        num_rows_total=num_rows_total,
        num_rows_per_page=MAX_ROWS,
    )


def create_rows_endpoint(
    processing_graph: ProcessingGraph,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    parquet_metadata_directory: StrPath,
    cache_max_days: int,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_key: Optional[str] = None,
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
    indexer = Indexer(
        processing_graph=processing_graph,
        hf_token=hf_token,
        parquet_metadata_directory=parquet_metadata_directory,
        httpfs=HTTPFileSystem(headers={"authorization": f"Bearer {hf_token}"}),
        unsupported_features=UNSUPPORTED_FEATURES,
        all_columns_supported_datasets_allow_list=ALL_COLUMNS_SUPPORTED_DATASETS_ALLOW_LIST,
    )

    async def rows_endpoint(request: Request) -> Response:
        await indexer.httpfs.set_session()
        revision: Optional[str] = None
        with StepProfiler(method="rows_endpoint", step="all"):
            try:
                with StepProfiler(method="rows_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    if not dataset or not config or not split or not are_valid_parameters([dataset, config, split]):
                        raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Offset must be positive")
                    length = int(request.query_params.get("length", MAX_ROWS))
                    if length < 0:
                        raise InvalidParameterError("Length must be positive")
                    if length > MAX_ROWS:
                        raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")
                    logging.info(
                        f"/rows, dataset={dataset}, config={config}, split={split}, offset={offset}, length={length}"
                    )
                with StepProfiler(method="rows_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
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
                except CachedArtifactError:
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
                        )
                with StepProfiler(method="rows_endpoint", step="query the rows"):
                    pa_table = rows_index.query(offset=offset, length=length)
                with StepProfiler(method="rows_endpoint", step="clean cache"):
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
                with StepProfiler(method="rows_endpoint", step="transform to a list"):
                    response = create_response(
                        dataset=dataset,
                        config=config,
                        split=split,
                        cached_assets_base_url=cached_assets_base_url,
                        cached_assets_directory=cached_assets_directory,
                        pa_table=pa_table,
                        offset=offset,
                        features=rows_index.parquet_index.features,
                        unsupported_columns=rows_index.parquet_index.unsupported_columns,
                        num_rows_total=rows_index.parquet_index.num_rows_total,
                    )
                with StepProfiler(method="rows_endpoint", step="update last modified time of rows in asset dir"):
                    update_last_modified_date_of_rows_in_assets_dir(
                        dataset=dataset,
                        config=config,
                        split=split,
                        offset=offset,
                        length=length,
                        assets_directory=cached_assets_directory,
                    )
                with StepProfiler(method="rows_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiError) else UnexpectedApiError("Unexpected error.", e)
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return rows_endpoint
