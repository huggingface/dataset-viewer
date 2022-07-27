from http import HTTPStatus
from typing import Any, Literal, Optional

from libutils.exceptions import CustomError
from libutils.utils import orjson_dumps
from starlette.responses import JSONResponse, Response

from api.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS

ApiErrorCode = Literal[
    "MissingRequiredParameter",
    "SplitsResponseNotReady",
    "FirstRowsResponseNotReady",
    "SplitsResponseNotFound",
    "FirstRowsResponseNotFound",
    "UnexpectedError",
]


class ApiCustomError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self, message: str, code: ApiErrorCode, status_code: HTTPStatus, cause: Optional[BaseException] = None
    ):
        super().__init__(message, str(code), cause)
        self.status_code = status_code


class MissingRequiredParameterError(ApiCustomError):
    """Raised when a required parameter is missing."""

    def __init__(self, message: str):
        super().__init__(message, "MissingRequiredParameter", HTTPStatus.UNPROCESSABLE_ENTITY, None)


class SplitsResponseNotReadyError(ApiCustomError):
    """Raised when the /splits response has not been processed yet."""

    def __init__(self, message: str):
        super().__init__(message, "SplitsResponseNotReady", HTTPStatus.BAD_GATEWAY, None)


class FirstRowsResponseNotReadyError(ApiCustomError):
    """Raised when the /first-rows response has not been processed yet."""

    def __init__(self, message: str):
        super().__init__(message, "FirstRowsResponseNotReady", HTTPStatus.BAD_GATEWAY, None)


class FirstRowsResponseNotFoundError(ApiCustomError):
    """Raised when the response for /first-rows has not been found."""

    def __init__(self, message: str):
        super().__init__(message, "FirstRowsResponseNotFound", HTTPStatus.NOT_FOUND, None)


class SplitsResponseNotFoundError(ApiCustomError):
    """Raised when the response for /splits has not been found."""

    def __init__(self, message: str):
        super().__init__(message, "SplitsResponseNotFound", HTTPStatus.NOT_FOUND, None)


class UnexpectedError(ApiCustomError):
    """Raised when the response for the split has not been found."""

    def __init__(self, message: str):
        super().__init__(message, "UnexpectedError", HTTPStatus.INTERNAL_SERVER_ERROR, None)


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


def get_json_ok_response(content: Any) -> Response:
    return get_json_response(content, max_age=MAX_AGE_LONG_SECONDS)


def get_json_error_response(
    content: Any, status_code: HTTPStatus = HTTPStatus.OK, error_code: Optional[str] = None
) -> Response:
    return get_json_response(content, status_code=status_code, max_age=MAX_AGE_SHORT_SECONDS, error_code=error_code)


def get_json_api_error_response(error: ApiCustomError) -> Response:
    return get_json_error_response(error.as_response_without_cause(), error.status_code, error.code)
