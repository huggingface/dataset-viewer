# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys
import traceback
from http import HTTPStatus
from typing import List, Literal, Optional, TypedDict, Union


class ErrorResponseWithoutCause(TypedDict):
    error: str


class ErrorResponseWithCause(ErrorResponseWithoutCause, total=False):
    cause_exception: str
    cause_message: str
    cause_traceback: List[str]


ErrorResponse = Union[ErrorResponseWithoutCause, ErrorResponseWithCause]


class LoggedError(Exception):
    def __init__(self, message: str):
        self.message = message
        logging.debug(self.message)
        super().__init__(self.message)


class CustomError(LoggedError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: str,
        cause: Optional[BaseException] = None,
        disclose_cause: Optional[bool] = None,
    ):
        super().__init__(message)
        self.exception = type(self).__name__
        self.status_code = status_code
        self.code = code
        self.message = str(self)
        self.disclose_cause = disclose_cause if disclose_cause is not None else cause is not None
        if cause is not None:
            self.cause_exception: Optional[str] = type(cause).__name__
            self.cause_message: Optional[str] = str(cause)
            (t, v, tb) = sys.exc_info()
            self.cause_traceback: Optional[List[str]] = traceback.format_exception(t, v, tb)
        else:
            self.cause_exception = None
            self.cause_message = None
            self.cause_traceback = None

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


CacheableErrorCode = Literal[
    "ConfigNamesError",
    "DatasetInBlockListError",
    "DatasetInfoHubRequestError",
    "DatasetManualDownloadError",
    "DatasetModuleNotInstalledError",
    "DatasetNotFoundError",
    "DatasetRevisionEmptyError",
    "DatasetRevisionNotFoundError",
    "DatasetTooBigFromDatasetsError",
    "DatasetTooBigFromHubError",
    "DatasetWithTooBigExternalFilesError",
    "DatasetWithTooManyExternalFilesError",
    "DisabledViewerError",
    "EmptyDatasetError",
    "ExternalFilesSizeRequestConnectionError",
    "ExternalFilesSizeRequestError",
    "ExternalFilesSizeRequestHTTPError",
    "ExternalFilesSizeRequestTimeoutError",
    "ExternalServerError",
    "FeaturesError",
    "FileSystemError",
    "InfoError",
    "JobManagerCrashedError",
    "JobManagerExceededMaximumDurationError",
    "MissingSpawningTokenError",
    "NormalRowsError",
    "ParameterMissingError",
    "ParquetResponseEmptyError",
    "PreviousStepFormatError",
    "PreviousStepStatusError",
    "ResponseAlreadyComputedError",
    "RowsPostProcessingError",
    "SplitsNamesError",
    "SplitNamesFromStreamingError",
    "SplitNotFoundError",
    "StreamingRowsError",
    "TooBigContentError",
    "TooManyColumnsError",
    "UnexpectedError",
    "UnsupportedExternalFilesError",
]


class CacheableError(CustomError):
    """Base class for exceptions that can be cached in the database."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: CacheableErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class ConfigNamesError(CacheableError):
    """Raised when the config names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ConfigNamesError", cause, True)


class DatasetInBlockListError(CacheableError):
    """Raised when the dataset is in the list of blocked datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetInBlockListError", cause, False)


class DatasetInfoHubRequestError(CacheableError):
    """Raised when the request to the Hub's dataset-info endpoint times out."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="DatasetInfoHubRequestError",
            cause=cause,
            disclose_cause=False,
        )


class DatasetManualDownloadError(CacheableError):
    """Raised when the dataset requires manual download."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetManualDownloadError", cause, True)


class DatasetModuleNotInstalledError(CacheableError):
    """Raised when the dataset tries to import a module that is not installed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetModuleNotInstalledError", cause, True)


class DatasetNotFoundError(CacheableError):
    """Raised when the dataset does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="DatasetNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class DatasetRevisionEmptyError(CacheableError):
    """Raised when the current git revision (branch, commit) could not be obtained."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetRevisionEmptyError", cause, False)


class DatasetRevisionNotFoundError(CacheableError):
    """Raised when the revision of a dataset repository does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "DatasetRevisionNotFoundError", cause, False)


class DatasetTooBigFromDatasetsError(CacheableError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromDatasetsError", cause, False)


class DatasetTooBigFromHubError(CacheableError):
    """Raised when the dataset size (sum of files on the Hub) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetTooBigFromHubError", cause, False)


class DatasetWithTooBigExternalFilesError(CacheableError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooBigExternalFilesError", cause, True)


class DatasetWithTooManyExternalFilesError(CacheableError):
    """Raised when the dataset size (sum of config sizes given by the datasets library) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManyExternalFilesError", cause, True)


class DisabledViewerError(CacheableError):
    """Raised when the dataset viewer is disabled."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="DisabledViewerError",
            cause=cause,
            disclose_cause=False,
        )


class EmptyDatasetError(CacheableError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class ExternalFilesSizeRequestConnectionError(CacheableError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestConnectionError", cause, True)


class ExternalFilesSizeRequestError(CacheableError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestError", cause, True)


class ExternalFilesSizeRequestHTTPError(CacheableError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestHTTPError", cause, True)


class ExternalFilesSizeRequestTimeoutError(CacheableError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestTimeoutError", cause, True)


class ExternalServerError(CacheableError):
    """Raised when the spawning.ai server is not responding."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ExternalServerError", cause, False)


class FeaturesError(CacheableError):
    """Raised when the features could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesError", cause, True)


class FileSystemError(CacheableError):
    """Raised when an error happen reading from File System."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FileSystemError", cause, False)


class InfoError(CacheableError):
    """Raised when the info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class JobManagerCrashedError(CacheableError):
    """Raised when the job runner crashed and the job became a zombie."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobManagerCrashedError",
            cause=cause,
            disclose_cause=False,
        )


class JobManagerExceededMaximumDurationError(CacheableError):
    """Raised when the job runner was killed because the job exceeded the maximum duration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobManagerExceededMaximumDurationError",
            cause=cause,
            disclose_cause=False,
        )


class MissingSpawningTokenError(CacheableError):
    """Raised when the spawning.ai token is not set."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingSpawningTokenError", cause, False)


class NormalRowsError(CacheableError):
    """Raised when the rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)


class ParameterMissingError(CacheableError):
    """Raised when request is missing some parameter."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
            code="ParameterMissingError",
            cause=cause,
            disclose_cause=False,
        )


class ParquetResponseEmptyError(CacheableError):
    """Raised when no parquet files were found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ParquetResponseEmptyError", cause, False)


class PreviousStepFormatError(CacheableError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class PreviousStepStatusError(CacheableError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class ResponseAlreadyComputedError(CacheableError):
    """Raised when response has been already computed by another job runner."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="ResponseAlreadyComputedError",
            cause=cause,
            disclose_cause=True,
        )


class RowsPostProcessingError(CacheableError):
    """Raised when the rows could not be post-processed successfully."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "RowsPostProcessingError", cause, False)


class SplitsNamesError(CacheableError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class SplitNamesFromStreamingError(CacheableError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitNamesFromStreamingError", cause, True)


class SplitNotFoundError(CacheableError):
    """Raised when the split does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="SplitNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class StreamingRowsError(CacheableError):
    """Raised when the rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class TooBigContentError(CacheableError):
    """Raised when content size in bytes is bigger than the supported value."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="TooBigContentError",
            cause=cause,
            disclose_cause=False,
        )


class TooManyColumnsError(CacheableError):
    """Raised when the dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class UnexpectedError(CacheableError):
    """Raised when the job runner raised an unexpected error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="UnexpectedError",
            cause=cause,
            disclose_cause=False,
        )
        logging.error(message, exc_info=cause)


class UnsupportedExternalFilesError(CacheableError):
    """Raised when we failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "UnsupportedExternalFilesError", cause, True)
