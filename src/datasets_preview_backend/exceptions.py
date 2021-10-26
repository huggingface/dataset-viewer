from typing import Optional, TypedDict, Union


class StatusErrorContent(TypedDict):
    status_code: int
    exception: str
    message: str
    cause: Union[str, None]
    cause_message: Union[str, None]


class StatusError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message: str, status_code: int, cause: Optional[Exception] = None):
        super().__init__(message)
        self.status_code = status_code
        self.exception = type(self).__name__
        self.message = str(self)
        self.cause_name = None if cause is None else type(cause).__name__
        self.cause_message = None if cause is None else str(cause)

    def as_content(self) -> StatusErrorContent:
        return {
            "status_code": self.status_code,
            "exception": self.exception,
            "message": self.message,
            "cause": self.cause_name,
            "cause_message": self.cause_message,
        }


class Status400Error(StatusError):
    """Exception raised if the response must be a 400 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, 400, cause)


class Status404Error(StatusError):
    """Exception raised if the response must be a 404 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, 404, cause)
