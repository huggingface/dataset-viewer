import sys
import traceback
from typing import List, Optional, TypedDict


class StatusErrorContent(TypedDict):
    status_code: int
    exception: str
    message: str
    cause_exception: str
    cause_message: str
    cause_traceback: List[str]


class StatusError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message: str, status_code: int, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.status_code = status_code
        self.exception = type(self).__name__
        self.message = str(self)
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


class Status500Error(StatusError):
    """Exception raised if the response must be a 500 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, 500, cause)
