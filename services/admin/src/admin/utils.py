# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Callable, Coroutine, List, Literal, Optional

from libcommon.exceptions import CustomError
from libcommon.utils import orjson_dumps
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

AdminErrorCode = Literal[
    "MissingRequiredParameter",
    "InvalidParameter",
    "UnsupportedDatasetError",
    "UnexpectedError",
    "ExternalUnauthenticatedError",
    "ExternalAuthenticatedError",
]


class AdminCustomError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: AdminErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class MissingRequiredParameterError(AdminCustomError):
    """Raised when a required parameter is missing."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNPROCESSABLE_ENTITY, "MissingRequiredParameter")


class InvalidParameterError(AdminCustomError):
    """Raised when a parameter is invalid."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNPROCESSABLE_ENTITY, "InvalidParameter")


class UnsupportedDatasetError(AdminCustomError):
    """Raised when a dataset is not supported (private dataset, for example)."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "UnsupportedDatasetError")


class UnexpectedError(AdminCustomError):
    """Raised when an unexpected error occurred."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "UnexpectedError")


class ExternalUnauthenticatedError(AdminCustomError):
    """Raised when the external authentication check failed while the user was unauthenticated."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "ExternalUnauthenticatedError")


class ExternalAuthenticatedError(AdminCustomError):
    """Raised when the external authentication check failed while the user was authenticated."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ExternalAuthenticatedError")


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson_dumps(content)


def get_response(content: Any, status_code: int = 200, max_age: int = 0) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}"} if max_age > 0 else {"Cache-Control": "no-store"}
    return OrjsonResponse(content, status_code=status_code, headers=headers)


def get_json_response(
    content: Any, status_code: HTTPStatus = HTTPStatus.OK, max_age: int = 0, error_code: Optional[str] = None
) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}" if max_age > 0 else "no-store"}
    if error_code is not None:
        headers["X-Error-Code"] = error_code
    return OrjsonResponse(content, status_code=status_code.value, headers=headers)


def get_json_ok_response(content: Any, max_age: int) -> Response:
    return get_json_response(content=content, max_age=max_age)


def get_json_error_response(
    content: Any, max_age: int, status_code: HTTPStatus = HTTPStatus.OK, error_code: Optional[str] = None
) -> Response:
    return get_json_response(content=content, status_code=status_code, max_age=max_age, error_code=error_code)


def get_json_admin_error_response(error: AdminCustomError, max_age: int) -> Response:
    return get_json_error_response(
        content=error.as_response(), status_code=error.status_code, max_age=max_age, error_code=error.code
    )


def is_non_empty_string(string: Any) -> bool:
    return isinstance(string, str) and bool(string and string.strip())


def are_valid_parameters(parameters: List[Any]) -> bool:
    return all(is_non_empty_string(s) for s in parameters)


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]
