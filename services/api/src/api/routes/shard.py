# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libapi.authentication import auth_check
from libapi.exceptions import (
    InvalidParameterError,
    MissingRequiredParameterError,
    ResponseNotFoundError,
    ResponseNotReadyError,
    UnexpectedApiError,
)
from libapi.request import get_request_parameter
from libapi.shard_utils import get_shard_info
from libapi.utils import (
    Endpoint,
    get_cache_entry_from_step,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)
from libcommon.exceptions import NotSupportedError
from libcommon.prometheus import StepProfiler
from libcommon.storage_client import StorageClient
from starlette.requests import Request
from starlette.responses import Response


def create_shard_endpoint(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    storage_clients: Optional[list[StorageClient]] = None,
) -> Endpoint:
    """Create the /shard endpoint handler."""

    async def shard_endpoint(request: Request) -> Response:
        method = "shard_endpoint"
        revision: Optional[str] = None
        with StepProfiler(method=method, step="all"):
            try:
                # 1. Validate parameters
                with StepProfiler(method=method, step="validate parameters"):
                    dataset = get_request_parameter(request, "dataset")
                    config = get_request_parameter(request, "config")
                    split = get_request_parameter(request, "split")
                    row_str = get_request_parameter(request, "row")

                    if not dataset:
                        raise MissingRequiredParameterError("Parameter 'dataset' is required")
                    if not config:
                        raise MissingRequiredParameterError("Parameter 'config' is required")
                    if not split:
                        raise MissingRequiredParameterError("Parameter 'split' is required")
                    if not row_str:
                        raise MissingRequiredParameterError("Parameter 'row' is required")

                    try:
                        row = int(row_str)
                    except ValueError:
                        raise InvalidParameterError("Invalid 'row' parameter - must be integer")

                    logging.debug(f"/shard {dataset=} {config=} {split=} {row=}")

                # 2. Auth check
                with StepProfiler(method=method, step="check authentication"):
                    await auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )

                # 3. Fetch cached data (single call - optimization)
                with StepProfiler(method=method, step="get cache entry"):
                    # Use config-parquet-and-info which has BOTH dataset_info AND parquet_files
                    result = get_cache_entry_from_step(
                        processing_step_name="config-parquet-and-info",
                        dataset=dataset,
                        config=config,
                        split=None,  # config-level cache
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        blocked_datasets=blocked_datasets,
                        hf_timeout_seconds=hf_timeout_seconds,
                        storage_clients=storage_clients,
                    )
                    revision = result.get("dataset_git_revision")

                    if result["http_status"] != HTTPStatus.OK:
                        return get_json_error_response(
                            content=result["content"],
                            status_code=result["http_status"],
                            max_age=max_age_short,
                            error_code=result["error_code"],
                            revision=revision,
                        )

                # 4. Extract split info
                with StepProfiler(method=method, step="extract split info"):
                    content = result["content"]
                    dataset_info = content.get("dataset_info", {})
                    splits = dataset_info.get("splits", {})

                    if split not in splits:
                        return get_json_error_response(
                            content={"error": f"Split '{split}' not found in config '{config}'"},
                            status_code=HTTPStatus.NOT_FOUND,
                            max_age=max_age_short,
                            error_code="SplitNotFound",
                            revision=revision,
                        )

                    split_info = splits[split]
                    parquet_files = content.get("parquet_files", [])

                # 5. Compute shard info
                with StepProfiler(method=method, step="compute shard info"):
                    try:
                        shard_result = get_shard_info(
                            row_index=row,
                            split_info=split_info,
                            parquet_files=parquet_files,
                            split=split,
                        )
                    except IndexError as e:
                        return get_json_error_response(
                            content={"error": str(e)},
                            status_code=HTTPStatus.BAD_REQUEST,
                            max_age=max_age_short,
                            error_code="RowOutOfBounds",
                            revision=revision,
                        )
                    except ValueError as e:
                        return get_json_error_response(
                            content={"error": str(e)},
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            max_age=max_age_short,
                            error_code="MetadataCorrupted",
                            revision=revision,
                        )

                # 6. Return success
                with StepProfiler(method=method, step="generate OK response"):
                    return get_json_ok_response(
                        content=shard_result,
                        max_age=max_age_long,
                        revision=revision,
                    )

            except Exception as e:
                error = (
                    e
                    if isinstance(
                        e,
                        (
                            InvalidParameterError,
                            MissingRequiredParameterError,
                            NotSupportedError,
                            ResponseNotFoundError,
                            ResponseNotReadyError,
                        ),
                    )
                    else UnexpectedApiError("Unexpected error.", e)
                )
                with StepProfiler(method=method, step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return shard_endpoint
