# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Callable, Coroutine, List, Literal, Optional

from libcommon.exceptions import CustomError
from libcommon.utils import orjson_dumps
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

ApiErrorCode = Literal[
    "MissingRequiredParameter",
    "ResponseNotReady",
    "ResponseNotFound",
    "UnexpectedError",
    "ExternalUnauthenticatedError",
    "ExternalAuthenticatedError",
]


class ApiCustomError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ApiErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class MissingRequiredParameterError(ApiCustomError):
    """Raised when a required parameter is missing."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNPROCESSABLE_ENTITY, "MissingRequiredParameter")


class ResponseNotReadyError(ApiCustomError):
    """Raised when the response has not been processed yet."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ResponseNotReady")


class ResponseNotFoundError(ApiCustomError):
    """Raised when the response has not been found."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ResponseNotFound")


class UnexpectedError(ApiCustomError):
    """Raised when the server raised an unexpected error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "UnexpectedError", cause)
        if cause:
            logging.exception(message, exc_info=cause)
        else:
            logging.exception(message)


class ExternalUnauthenticatedError(ApiCustomError):
    """Raised when the external authentication check failed while the user was unauthenticated."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "ExternalUnauthenticatedError")


class ExternalAuthenticatedError(ApiCustomError):
    """Raised when the external authentication check failed while the user was authenticated.

    Even if the external authentication server returns 403 in that case, we return 404 because
    we don't know if the dataset exist or not. It's also coherent with how the Hugging Face Hub works."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ExternalAuthenticatedError")


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson_dumps(content=content)


def get_response(content: Any, status_code: int = 200, max_age: int = 0) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}"} if max_age > 0 else {"Cache-Control": "no-store"}
    return OrjsonResponse(content=content, status_code=status_code, headers=headers)


def get_json_response(
    content: Any, status_code: HTTPStatus = HTTPStatus.OK, max_age: int = 0, error_code: Optional[str] = None
) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}" if max_age > 0 else "no-store"}
    if error_code is not None:
        headers["X-Error-Code"] = error_code
    return OrjsonResponse(content=content, status_code=status_code.value, headers=headers)


def get_json_ok_response(content: Any, max_age: int = 0) -> Response:
    return get_json_response(content=content, max_age=max_age)


def get_json_error_response(
    content: Any, status_code: HTTPStatus = HTTPStatus.OK, max_age: int = 0, error_code: Optional[str] = None
) -> Response:
    return get_json_response(content=content, status_code=status_code, max_age=max_age, error_code=error_code)


def get_json_api_error_response(error: ApiCustomError, max_age: int = 0) -> Response:
    return get_json_error_response(
        content=error.as_response(), status_code=error.status_code, max_age=max_age, error_code=error.code
    )


def is_non_empty_string(string: Any) -> bool:
    return isinstance(string, str) and bool(string and string.strip())


def are_valid_parameters(parameters: List[Any]) -> bool:
    return all(is_non_empty_string(s) for s in parameters)


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]
