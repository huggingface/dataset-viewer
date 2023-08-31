# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import os
import shutil
from http import HTTPStatus
from itertools import islice
from typing import Any, Callable, Coroutine, List, Optional

import pyarrow as pa
from datasets import Features
from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.orchestrator import DatasetOrchestrator
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.rows_utils import transform_rows
from libcommon.simple_cache import (
    CACHED_RESPONSE_NOT_FOUND,
    CacheEntry,
    get_best_response,
)
from libcommon.storage import StrPath
from libcommon.utils import Priority, RowItem, orjson_dumps
from libcommon.viewer_utils.asset import glob_rows_in_assets_dir
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from libapi.exceptions import (
    ResponseNotFoundError,
    ResponseNotReadyError,
    TransformRowsProcessingError,
)


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
) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}" if max_age > 0 else "no-store"}
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


def get_json_ok_response(content: Any, max_age: int = 0, revision: Optional[str] = None) -> Response:
    return get_json_response(content=content, max_age=max_age, revision=revision)


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
    return isinstance(string, str) and bool(string and string.strip())


def are_valid_parameters(parameters: List[Any]) -> bool:
    return all(is_non_empty_string(s) for s in parameters)


def try_backfill_dataset_then_raise(
    processing_steps: List[ProcessingStep],
    dataset: str,
    processing_graph: ProcessingGraph,
    cache_max_days: int,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    dataset_orchestrator = DatasetOrchestrator(dataset=dataset, processing_graph=processing_graph)
    if not dataset_orchestrator.has_some_cache():
        # We have to check if the dataset exists and is supported
        try:
            revision = get_dataset_git_revision(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                hf_timeout_seconds=hf_timeout_seconds,
            )
        except Exception as e:
            # The dataset is not supported
            raise ResponseNotFoundError("Not found.") from e
        # The dataset is supported, and the revision is known. We set the revision (it will create the jobs)
        # and tell the user to retry.
        logging.info(f"Set orchestrator revision for dataset={dataset}, revision={revision}")
        dataset_orchestrator.set_revision(
            revision=revision, priority=Priority.NORMAL, error_codes_to_retry=[], cache_max_days=cache_max_days
        )
        raise ResponseNotReadyError(
            "The server is busier than usual and the response is not ready yet. Please retry later."
        )
    elif dataset_orchestrator.has_pending_ancestor_jobs(
        processing_step_names=[processing_step.name for processing_step in processing_steps]
    ):
        # some jobs are still in progress, the cache entries could exist in the future
        raise ResponseNotReadyError(
            "The server is busier than usual and the response is not ready yet. Please retry later."
        )
    else:
        # no pending job: the cache entry will not be created
        raise ResponseNotFoundError("Not found.")


def get_cache_entry_from_steps(
    processing_steps: List[ProcessingStep],
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    processing_graph: ProcessingGraph,
    cache_max_days: int,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> CacheEntry:
    """Gets the cache from the first successful step in the processing steps list.
    If no successful result is found, it will return the last one even if it's an error,
    Checks if job is still in progress by each processing step in case of no entry found.
    Raises:
        - [`~utils.ResponseNotFoundError`]
          if no result is found.
        - [`~utils.ResponseNotReadyError`]
          if the response is not ready yet.

    Returns: the cached record
    """
    kinds = [processing_step.cache_kind for processing_step in processing_steps]
    best_response = get_best_response(kinds=kinds, dataset=dataset, config=config, split=split)
    if "error_code" in best_response.response and best_response.response["error_code"] == CACHED_RESPONSE_NOT_FOUND:
        try_backfill_dataset_then_raise(
            processing_steps=processing_steps,
            processing_graph=processing_graph,
            dataset=dataset,
            hf_endpoint=hf_endpoint,
            hf_timeout_seconds=hf_timeout_seconds,
            hf_token=hf_token,
            cache_max_days=cache_max_days,
        )
    return best_response.response


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]


def to_rows_list(
    pa_table: pa.Table,
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
    features: Features,
    unsupported_columns: List[str],
    row_idx_column: Optional[str] = None,
) -> List[RowItem]:
    num_rows = pa_table.num_rows
    for idx, (column, feature) in enumerate(features.items()):
        if column in unsupported_columns:
            pa_table = pa_table.add_column(idx, column, pa.array([None] * num_rows))
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(
            dataset=dataset,
            config=config,
            split=split,
            rows=pa_table.to_pylist(),
            features=features,
            cached_assets_base_url=cached_assets_base_url,
            cached_assets_directory=cached_assets_directory,
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


def _greater_or_equal(row_dir_name: str, row_idx: int, on_error: bool) -> bool:
    try:
        return int(row_dir_name) >= row_idx
    except ValueError:
        return on_error


def clean_cached_assets(
    dataset: str,
    cached_assets_directory: StrPath,
    keep_first_rows_number: int,
    keep_most_recent_rows_number: int,
    max_cleaned_rows_number: int,
) -> None:
    """
    The cached assets directory is cleaned to save disk space using this simple (?) heuristic:

    1. it takes a big sample of rows from the cache using glob (max `max_cleaned_rows_number`)
    2. it keeps the most recent ones (max `keep_most_recent_rows_number`)
    3. it keeps the rows below a certain index (max `keep_first_rows_number`)
    4. it discards the rest

    To check for the most recent rows, it looks at the "last modified time" of rows directories.
    This time is updated every time a row is accessed using `update_last_modified_date_of_rows_in_assets_dir()`.

    Args:
        dataset (`str`):
            Dataset name e.g `squad` or `lhoestq/demo1`.
            Rows are cleaned in any dataset configuration or split of this dataset.
        cached_assets_directory (`str`):
            Directory containing the cached image and audio files
        keep_first_rows_number (`int`):
            Keep the rows with an index below a certain number
        keep_most_recent_rows_number (`int`):
            Keep the most recently accessed rows.
        max_cleaned_rows_number (`int`):
            Maximum number of rows to discard.
    """
    if keep_first_rows_number < 0 or keep_most_recent_rows_number < 0 or max_cleaned_rows_number < 0:
        raise ValueError(
            "Failed to run cached assets cleaning. Make sure all of keep_first_rows_number,"
            f" keep_most_recent_rows_number and max_cleaned_rows_number  are set (got {keep_first_rows_number},"
            f" {keep_most_recent_rows_number} and {max_cleaned_rows_number})"
        )
    row_directories = glob_rows_in_assets_dir(dataset, cached_assets_directory)
    row_directories_sample = list(
        islice(
            (
                row_dir
                for row_dir in row_directories
                if _greater_or_equal(row_dir.name, keep_first_rows_number, on_error=True)
            ),
            max_cleaned_rows_number + keep_most_recent_rows_number,
        )
    )
    if len(row_directories_sample) > keep_most_recent_rows_number:
        row_dirs_to_delete = sorted(row_directories_sample, key=os.path.getmtime, reverse=True)[
            keep_most_recent_rows_number:
        ]
        for row_dir_to_delete in row_dirs_to_delete:
            shutil.rmtree(row_dir_to_delete, ignore_errors=True)
