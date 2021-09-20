import logging

from datasets_preview_backend.types import StatusErrorDict


class StatusError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)
        # TODO: use caller's __name__ instead of this file's name
        logger = logging.getLogger(__name__)
        logger.warning(f"Error {self.status_code} '{self.message}'.")
        logger.debug(f"Caused by a {type(self.__cause__).__name__}: '{str(self.__cause__)}'")

    def as_dict(self) -> StatusErrorDict:
        return {
            "status_code": self.status_code,
            "exception": type(self).__name__,
            "message": str(self),
            "cause": type(self.__cause__).__name__,
            "cause_message": str(self.__cause__),
        }


class Status400Error(StatusError):
    """Exception raised if the response must be a 400 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str):
        super().__init__(message, 400)


class Status404Error(StatusError):
    """Exception raised if the response must be a 404 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message: str):
        super().__init__(message, 404)
