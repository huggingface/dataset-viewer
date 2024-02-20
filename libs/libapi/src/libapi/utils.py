# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from collections.abc import Callable, Coroutine
from http import HTTPStatus
from typing import Any, Optional

import pyarrow as pa
from datasets import Features
from libcommon.dtos import Priority, RowItem
from libcommon.exceptions import CustomError
from libcommon.operations import update_dataset
from libcommon.orchestrator import has_pending_ancestor_jobs
from libcommon.simple_cache import CACHED_RESPONSE_NOT_FOUND, CacheEntry, get_response_or_missing_error, has_some_cache
from libcommon.storage_client import StorageClient
from libcommon.utils import orjson_dumps
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from libapi.exceptions import (
    ResponseNotFoundError,
    ResponseNotReadyError,
    TransformRowsProcessingError,
)
from libapi.rows_utils import transform_rows


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson_dumps(content=content)


def get_response(content: Any, status_code: int = 200, max_age: int = 0) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}"} if max_age > 0 else {"Cache-Control": "no-store"}
    return OrjsonResponse(content=content, status_code=status_code, headers=headers)


def get_json_response(
    content: Any,
    status_code: HTTPStatus = HTTPStatus.OK,
    max_age: int = 0,
    error_code: Optional[str] = None,
    revision: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> Response:
    if not headers:
        headers = {}
    headers["Cache-Control"] = f"max-age={max_age}" if max_age > 0 else "no-store"
    if error_code is not None:
        headers["X-Error-Code"] = error_code
    if revision is not None:
        headers["X-Revision"] = revision
    return OrjsonResponse(content=content, status_code=status_code.value, headers=headers)


# these headers are exposed to the client (browser)
EXPOSED_HEADERS = [
    "X-Error-Code",
    "X-Revision",
]


def get_json_ok_response(
    content: Any, max_age: int = 0, revision: Optional[str] = None, headers: Optional[dict[str, str]] = None
) -> Response:
    return get_json_response(content=content, max_age=max_age, revision=revision, headers=headers)


def get_json_error_response(
    content: Any,
    status_code: HTTPStatus = HTTPStatus.OK,
    max_age: int = 0,
    error_code: Optional[str] = None,
    revision: Optional[str] = None,
) -> Response:
    return get_json_response(
        content=content, status_code=status_code, max_age=max_age, error_code=error_code, revision=revision
    )


def get_json_api_error_response(error: CustomError, max_age: int = 0, revision: Optional[str] = None) -> Response:
    return get_json_error_response(
        content=error.as_response(),
        status_code=error.status_code,
        max_age=max_age,
        error_code=error.code,
        revision=revision,
    )


def is_non_empty_string(string: Any) -> bool:
    return isinstance(string, str) and bool(string.strip())


def are_valid_parameters(parameters: list[Any]) -> bool:
    return all(is_non_empty_string(s) for s in parameters)


def try_backfill_dataset_then_raise(
    processing_step_names: list[str],
    dataset: str,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    """
    Tries to backfill the dataset, and then raises an error.
    """
    if has_pending_ancestor_jobs(dataset=dataset, processing_step_names=processing_step_names):
        logging.debug("Cache entry not found but some jobs are still in progress, so it could exist in the future")
        raise ResponseNotReadyError(
            "The server is busier than usual and the response is not ready yet. Please retry later."
        )
    logging.debug("No pending job that could create the expected cache entry")
    if has_some_cache(dataset=dataset):
        logging.debug(
            "Some cache entries exist, so the dataset is supported, but that cache entry will never be created"
        )
        raise ResponseNotFoundError("Not found.")
    logging.debug("No cache entry found")
    update_dataset(
        dataset=dataset,
        blocked_datasets=blocked_datasets,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        hf_timeout_seconds=hf_timeout_seconds,
        priority=Priority.NORMAL,
        storage_clients=storage_clients,
    )
    # ^ raises with NotSupportedError if the dataset is not supported - in which case it's deleted from the cache
    logging.debug("The dataset is supported and it's being backfilled")
    raise ResponseNotReadyError(
        "The server is busier than usual and the response is not ready yet. Please retry later."
    )


def get_cache_entry_from_step(
    processing_step_name: str,
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> CacheEntry:
    """Gets the cache from a processing step.
    Checks if job is still in progress if the entry has not been found.

    Args:
        processing_step_name (`str`): the name of the processing step
        dataset (`str`): the dataset name
        config (`str`, *optional*): the config name
        split (`str`, *optional*): the split name
        hf_endpoint (`str`): the Hub endpoint
        blocked_datasets (`list[str]`): the list of blocked datasets
        hf_token (`str`, *optional*): the Hugging Face token
        hf_timeout_seconds (`float`, *optional*): the Hugging Face timeout in seconds
        storage_clients (`list[StorageClient]`, *optional*): the list of storage clients

    Raises:
        [~`utils.ResponseNotFoundError`]:
          if no result is found.
        [~`utils.ResponseNotReadyError`]:
          if the response is not ready yet.

    Returns:
        `CacheEntry`: the cached record
    """
    response = get_response_or_missing_error(kind=processing_step_name, dataset=dataset, config=config, split=split)
    if "error_code" in response and response["error_code"] == CACHED_RESPONSE_NOT_FOUND:
        try_backfill_dataset_then_raise(
            processing_step_names=[processing_step_name],
            dataset=dataset,
            hf_endpoint=hf_endpoint,
            blocked_datasets=blocked_datasets,
            hf_timeout_seconds=hf_timeout_seconds,
            hf_token=hf_token,
            storage_clients=storage_clients,
        )
    return response


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]


async def to_rows_list(
    pa_table: pa.Table,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    storage_client: StorageClient,
    row_idx_column: Optional[str] = None,
) -> list[RowItem]:
    num_rows = pa_table.num_rows
    for idx, (column, feature) in enumerate(features.items()):
        if column in unsupported_columns:
            pa_table = pa_table.add_column(idx, column, pa.array([None] * num_rows))
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = await transform_rows(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            rows=pa_table.to_pylist(),
            features=features,
            storage_client=storage_client,
            offset=offset,
            row_idx_column=row_idx_column,
        )
    except Exception as err:
        raise TransformRowsProcessingError(
            "Server error while post-processing the split rows. Please report the issue."
        ) from err
    return [
        {
            "row_idx": idx + offset if row_idx_column is None else row.pop(row_idx_column),
            "row": row,
            "truncated_cells": [],
        }
        for idx, row in enumerate(transformed_rows)
    ]
