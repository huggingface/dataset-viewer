from http import HTTPStatus
from typing import Literal, Optional

from libcommon.exceptions import (
    CustomError,
    ErrorResponseWithCause,
    ErrorResponseWithoutCause,
)
from libcommon.simple_cache import CacheEntryWithDetails
from libcommon.utils import orjson_dumps

GeneralJobRunnerErrorCode = Literal[
    "ParameterMissingError",
    "NoGitRevisionError",
    "SplitNotFoundError",
    "UnexpectedError",
    "TooBigContentError",
    "JobManagerCrashedError",
    "JobManagerExceededMaximumDurationError",
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
    """Raised when response has been already computed by another job runner."""

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


class JobManagerCrashedError(GeneralJobRunnerError):
    """Raised when the job runner crashed and the job became a zombie."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobManagerCrashedError",
            cause=cause,
            disclose_cause=False,
        )


class JobManagerExceededMaximumDurationError(GeneralJobRunnerError):
    """Raised when the job runner was killed because the job exceeded the maximum duration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobManagerExceededMaximumDurationError",
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


class PreviousStepError(JobRunnerError):
    """Raised when the previous step failed. It contains the contents of the error response,
    and the details contain extra information about the previous step.
    """

    error_with_cause: ErrorResponseWithCause
    error_without_cause: ErrorResponseWithoutCause

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: str,
        cause: Optional[BaseException],
        disclose_cause: bool,
        error_with_cause: ErrorResponseWithCause,
        error_without_cause: ErrorResponseWithoutCause,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )
        self.error_with_cause = error_with_cause
        self.error_without_cause = error_without_cause

    @staticmethod
    def from_response(
        response: CacheEntryWithDetails,
        kind: str,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> "PreviousStepError":
        if response.get("http_status") == HTTPStatus.OK:
            raise ValueError("Cannot create a PreviousStepError, the response should contain an error")

        message = response["content"]["error"] if "error" in response["content"] else "Unknown error"
        status_code = response["http_status"]
        error_code = response["error_code"] or "PreviousStepError"
        cause = None  # No way to create the same exception
        disclose_cause = orjson_dumps(response["details"]) == orjson_dumps(response["content"])
        error_without_cause: ErrorResponseWithoutCause = {"error": message}
        error_with_cause: ErrorResponseWithCause = {
            "error": message,
            # Add lines in the traceback to give some info about the previous step error (a bit hacky)
            "cause_traceback": [
                "The previous step failed, the error is copied to this step:",
                f"  {kind=} {dataset=} {config=} {split=}",
                "---",
            ],
        }
        if "cause_exception" in response["details"] and isinstance(response["details"]["cause_exception"], str):
            error_with_cause["cause_exception"] = response["details"]["cause_exception"]
        if "cause_message" in response["details"] and isinstance(response["details"]["cause_message"], str):
            error_with_cause["cause_message"] = response["details"]["cause_message"]
        if (
            "cause_traceback" in response["details"]
            and isinstance(response["details"]["cause_traceback"], list)
            and all(isinstance(line, str) for line in response["details"]["cause_traceback"])
        ):
            error_with_cause["cause_traceback"].extend(response["details"]["cause_traceback"])
        return PreviousStepError(
            message=message,
            status_code=status_code,
            code=error_code,
            cause=cause,
            disclose_cause=disclose_cause,
            error_without_cause=error_without_cause,
            error_with_cause=error_with_cause,
        )

    def as_response_with_cause(self) -> ErrorResponseWithCause:
        return self.error_with_cause

    def as_response_without_cause(self) -> ErrorResponseWithoutCause:
        return self.error_without_cause
