from http import HTTPStatus
from typing import Any, Literal, Optional

from libutils.exceptions import CustomError
from libutils.utils import orjson_dumps
from starlette.responses import JSONResponse, Response

from admin.config import MAX_AGE_SHORT_SECONDS

AdminErrorCode = Literal[
    "InvalidParameter",
    "UnexpectedError",
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


class InvalidParameterError(AdminCustomError):
    """Raised when a parameter is invalid."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNPROCESSABLE_ENTITY, "InvalidParameter")


class UnexpectedError(AdminCustomError):
    """Raised when an unexpected error occurred."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "UnexpectedError")


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
    return get_json_response(content, max_age=MAX_AGE_SHORT_SECONDS)


def get_json_error_response(
    content: Any, status_code: HTTPStatus = HTTPStatus.OK, error_code: Optional[str] = None
) -> Response:
    return get_json_response(content, status_code=status_code, max_age=MAX_AGE_SHORT_SECONDS, error_code=error_code)


def get_json_admin_error_response(error: AdminCustomError) -> Response:
    return get_json_error_response(error.as_response(), error.status_code, error.code)
