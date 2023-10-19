# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys
import traceback
from http import HTTPStatus
from typing import Literal, Optional, TypedDict, Union


class ErrorResponseWithoutCause(TypedDict):
    error: str


class ErrorResponseWithCause(ErrorResponseWithoutCause, total=False):
    cause_exception: str
    cause_message: str
    cause_traceback: list[str]


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
            self.cause_traceback: Optional[list[str]] = traceback.format_exception(t, v, tb)
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
    "CacheDirectoryNotInitializedError",
    "ComputationError",
    "ConfigNamesError",
    "ConfigNotFoundError",
    "CreateCommitError",
    "DatasetInBlockListError",
    "DatasetInfoHubRequestError",
    "DatasetManualDownloadError",
    "DatasetModuleNotInstalledError",
    "DatasetNotFoundError",
    "DatasetRevisionEmptyError",
    "DatasetRevisionNotFoundError",
    "DatasetScriptError",
    "DatasetWithScriptNotSupportedError",
    "DatasetWithTooManyConfigsError",
    "DatasetWithTooManyParquetFilesError",
    "DisabledViewerError",
    "DiskError",
    "DuckDBIndexFileNotFoundError",
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
    "LockedDatasetTimeoutError",
    "MissingSpawningTokenError",
    "NoSupportedFeaturesError",
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
    "SplitWithTooBigParquetError",
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


class CacheDirectoryNotInitializedError(CacheableError):
    """The cache directory has not been initialized before job compute."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "CacheDirectoryNotInitializedError", cause, True)


class ConfigNamesError(CacheableError):
    """The config names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ConfigNamesError", cause, True)


class ConfigNotFoundError(CacheableError):
    """The config does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="ConfigNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class CreateCommitError(CacheableError):
    """A commit could not be created on the Hub."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "CreateCommitError", cause, False)


class DatasetInBlockListError(CacheableError):
    """The dataset is in the list of blocked datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetInBlockListError", cause, False)


class DatasetInfoHubRequestError(CacheableError):
    """The request to the Hub's dataset-info endpoint times out."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="DatasetInfoHubRequestError",
            cause=cause,
            disclose_cause=False,
        )


class DatasetManualDownloadError(CacheableError):
    """The dataset requires manual download."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetManualDownloadError", cause, True)


class DatasetModuleNotInstalledError(CacheableError):
    """The dataset tries to import a module that is not installed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetModuleNotInstalledError", cause, True)


class DatasetNotFoundError(CacheableError):
    """The dataset does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="DatasetNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class DatasetRevisionEmptyError(CacheableError):
    """The current git revision (branch, commit) could not be obtained."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetRevisionEmptyError", cause, False)


class DatasetRevisionNotFoundError(CacheableError):
    """The revision of a dataset repository does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "DatasetRevisionNotFoundError", cause, False)


class DatasetScriptError(CacheableError):
    """The dataset script generated an error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetScriptError", cause, False)


class DatasetWithTooManyConfigsError(CacheableError):
    """The number of configs of a dataset exceeded the limit."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManyConfigsError", cause, True)


class DatasetWithTooManyParquetFilesError(CacheableError):
    """The number of parquet files of a dataset is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManyParquetFilesError", cause, True)


class DuckDBIndexFileNotFoundError(CacheableError):
    """No duckdb index file was found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DuckDBIndexFileNotFoundError", cause, False)


class DisabledViewerError(CacheableError):
    """The dataset viewer is disabled."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="DisabledViewerError",
            cause=cause,
            disclose_cause=False,
        )


class DiskError(CacheableError):
    """Disk-related issues, for example, incorrect permissions."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="DiskError",
            cause=cause,
            disclose_cause=False,
        )


class EmptyDatasetError(CacheableError):
    """The dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class ExternalFilesSizeRequestConnectionError(CacheableError):
    """We failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestConnectionError", cause, True)


class ExternalFilesSizeRequestError(CacheableError):
    """We failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestError", cause, True)


class ExternalFilesSizeRequestHTTPError(CacheableError):
    """We failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestHTTPError", cause, True)


class ExternalFilesSizeRequestTimeoutError(CacheableError):
    """We failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "ExternalFilesSizeRequestTimeoutError", cause, True)


class ExternalServerError(CacheableError):
    """The spawning.ai server is not responding."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ExternalServerError", cause, False)


class FeaturesError(CacheableError):
    """The features could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesError", cause, True)


class FileSystemError(CacheableError):
    """An error happen reading from File System."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FileSystemError", cause, False)


class InfoError(CacheableError):
    """The info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class JobManagerCrashedError(CacheableError):
    """The job runner crashed and the job became a zombie."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobManagerCrashedError",
            cause=cause,
            disclose_cause=False,
        )


class JobManagerExceededMaximumDurationError(CacheableError):
    """The job runner was killed because the job exceeded the maximum duration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobManagerExceededMaximumDurationError",
            cause=cause,
            disclose_cause=False,
        )


class LockedDatasetTimeoutError(CacheableError):
    """A dataset is locked by another job."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "LockedDatasetTimeoutError", cause, True)


class MissingSpawningTokenError(CacheableError):
    """The spawning.ai token is not set."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingSpawningTokenError", cause, False)


class NormalRowsError(CacheableError):
    """The rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)


class NoSupportedFeaturesError(CacheableError):
    """The dataset does not have any features which types are supported by a worker's processing pipeline."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "NoSupportedFeaturesError", cause, True)


class ParameterMissingError(CacheableError):
    """The request is missing some parameter."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="ParameterMissingError",
            cause=cause,
            disclose_cause=False,
        )


class ParquetResponseEmptyError(CacheableError):
    """No parquet files were found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ParquetResponseEmptyError", cause, False)


class PreviousStepFormatError(CacheableError):
    """The content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class PreviousStepStatusError(CacheableError):
    """The previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class ResponseAlreadyComputedError(CacheableError):
    """The response has been already computed by another job runner."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="ResponseAlreadyComputedError",
            cause=cause,
            disclose_cause=True,
        )


class RowsPostProcessingError(CacheableError):
    """The rows could not be post-processed successfully."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "RowsPostProcessingError", cause, False)


class SplitsNamesError(CacheableError):
    """The split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class SplitNamesFromStreamingError(CacheableError):
    """The split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitNamesFromStreamingError", cause, True)


class SplitNotFoundError(CacheableError):
    """The split does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="SplitNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class SplitWithTooBigParquetError(CacheableError):
    """The split parquet size (sum of parquet sizes given) is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitWithTooBigParquetError", cause, False)


class StatisticsComputationError(CacheableError):
    """An unexpected behavior or error occurred during statistics computations."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ComputationError", cause, True)


class StreamingRowsError(CacheableError):
    """The rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class TooBigContentError(CacheableError):
    """The content size in bytes is bigger than the supported value."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="TooBigContentError",
            cause=cause,
            disclose_cause=False,
        )


class TooManyColumnsError(CacheableError):
    """The dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class UnexpectedError(CacheableError):
    """The job runner raised an unexpected error."""

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
    """We failed to get the size of the external files."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "UnsupportedExternalFilesError", cause, True)


class DatasetWithScriptNotSupportedError(CacheableError):
    """We don't support some datasets because they have a dataset script."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithScriptNotSupportedError", cause, True)
