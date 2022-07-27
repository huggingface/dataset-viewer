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
            self.cause_exception = type(cause).__name__
            self.cause_message = str(cause)
            (t, v, tb) = sys.exc_info()
            self.cause_traceback = traceback.format_exception(t, v, tb)
            self.disclose_cause = disclose_cause
        else:
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
        return self.as_response_without_cause() if self.disclose_cause else self.as_response_with_cause()


# to be deprecated
class StatusErrorContent(TypedDict):
    status_code: int
    exception: str
    message: str
    cause_exception: str
    cause_message: str
    cause_traceback: List[str]


class Status400ErrorResponse(TypedDict):
    error: str
    cause_exception: Optional[str]
    cause_message: Optional[str]
    cause_traceback: Optional[List[str]]


class Status500ErrorResponse(TypedDict):
    error: str


class StatusError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message: str, status_code: int, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.status_code = status_code
        self.exception = type(self).__name__
        self.message = str(self)
        # TODO: once /splits and /rows are deprecated, remove the conditional and as_content()
        if cause is None:
            self.cause_exception = self.exception
            self.cause_message = self.message
            self.cause_traceback = []
        else:
            self.cause_exception = type(cause).__name__
            self.cause_message = str(cause)
            (t, v, tb) = sys.exc_info()
            self.cause_traceback = traceback.format_exception(t, v, tb)

    def as_content(self) -> StatusErrorContent:
        return {
            "status_code": self.status_code,
            "exception": self.exception,
            "message": self.message,
            "cause_exception": self.cause_exception,
            "cause_message": self.cause_message,
            "cause_traceback": self.cause_traceback,
        }


class Status400Error(StatusError):
    """Exception raised if the response must be a 400 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, 400, cause)

    def as_response(self) -> Status400ErrorResponse:
        return {
            "error": self.message,
            # TODO: once /splits and /rows are deprecated, remove the conditionals
            "cause_exception": self.cause_exception if self.cause_message != self.message else None,
            "cause_message": self.cause_message if self.cause_message != self.message else None,
            "cause_traceback": self.cause_traceback if len(self.cause_traceback) else None,
        }


class Status500Error(StatusError):
    """Exception raised if the response must be a 500 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, 500, cause)

    def as_response(self) -> Status500ErrorResponse:
        return {
            "error": self.message,
        }
