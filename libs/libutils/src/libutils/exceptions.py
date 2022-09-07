import sys
import traceback
from http import HTTPStatus
from typing import List, Optional, TypedDict, Union


class ErrorResponseWithoutCause(TypedDict):
    error: str


class ErrorResponseWithCause(ErrorResponseWithoutCause, total=False):
    cause_exception: str
    cause_message: str
    cause_traceback: List[str]


ErrorResponse = Union[ErrorResponseWithoutCause, ErrorResponseWithCause]


class CustomError(Exception):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: str,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message)
        self.exception = type(self).__name__
        self.status_code = status_code
        self.code = code
        self.message = str(self)
        if cause is not None:
            self.cause_exception: Optional[str] = type(cause).__name__
            self.cause_message: Optional[str] = str(cause)
            (t, v, tb) = sys.exc_info()
            self.cause_traceback: Optional[List[str]] = traceback.format_exception(t, v, tb)
            self.disclose_cause = disclose_cause
        else:
            self.cause_exception = None
            self.cause_message = None
            self.cause_traceback = None
            self.disclose_cause = False

    def as_response_with_cause(self) -> ErrorResponseWithCause:
        error: ErrorResponseWithCause = {"error": self.message}
        if self.cause_exception is not None:
            error["cause_exception"] = self.cause_exception
        if self.cause_message is not None:
            error["cause_message"] = self.cause_message
        if self.cause_traceback is not None:
            error["cause_traceback"] = self.cause_traceback
        return error

    def as_response_without_cause(self) -> ErrorResponseWithoutCause:
        return {"error": self.message}

    def as_response(self) -> ErrorResponse:
        return self.as_response_with_cause() if self.disclose_cause else self.as_response_without_cause()
