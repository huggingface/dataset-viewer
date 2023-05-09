from http import HTTPStatus
from typing import Literal, Optional

from libcommon.exceptions import CustomError

GeneralJobRunnerErrorCode = Literal[
    "ParameterMissingError",
    "NoGitRevisionError",
    "SplitNotFoundError",
    "UnexpectedError",
    "TooBigContentError",
    "JobRunnerCrashedError",
    "JobRunnerExceededMaximumDurationError",
    "ResponseAlreadyComputedError",
]


class JobRunnerError(CustomError):
    """Base class for job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: str,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class GeneralJobRunnerError(JobRunnerError):
    """General class for job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: GeneralJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class SplitNotFoundError(GeneralJobRunnerError):
    """Raised when the split does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="SplitNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class ParameterMissingError(GeneralJobRunnerError):
    """Raised when request is missing some parameter."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
            code="ParameterMissingError",
            cause=cause,
            disclose_cause=False,
        )


class NoGitRevisionError(GeneralJobRunnerError):
    """Raised when the git revision returned by huggingface_hub is None."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="NoGitRevisionError",
            cause=cause,
            disclose_cause=False,
        )


class ResponseAlreadyComputedError(GeneralJobRunnerError):
    """Raised when response has been already computed by another operator."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ResponseAlreadyComputedError", cause, True)


class TooBigContentError(GeneralJobRunnerError):
    """Raised when content size in bytes is bigger than the supported value."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="TooBigContentError",
            cause=cause,
            disclose_cause=False,
        )


class UnexpectedError(GeneralJobRunnerError):
    """Raised when the job runner raised an unexpected error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="UnexpectedError",
            cause=cause,
            disclose_cause=False,
        )


class JobRunnerCrashedError(GeneralJobRunnerError):
    """Raised when the job runner crashed and the job became a zombie."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobRunnerCrashedError",
            cause=cause,
            disclose_cause=False,
        )


class JobRunnerExceededMaximumDurationError(GeneralJobRunnerError):
    """Raised when the job runner was killed because the job exceeded the maximum duration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobRunnerExceededMaximumDurationError",
            cause=cause,
            disclose_cause=False,
        )


class StreamingRowsError(JobRunnerError):
    """Raised when the rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class NormalRowsError(JobRunnerError):
    """Raised when the rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)
