# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable, Coroutine, List, Optional

from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.orchestrator import DatasetOrchestrator
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.utils import Priority, orjson_dumps
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from libapi.exceptions import ResponseNotFoundError, ResponseNotReadyError


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


def try_backfill_dataset(
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


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]
