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
    "DataFilesNotFoundError",
    "DatasetGenerationError",
    "DatasetGenerationCastError",
    "DatasetInBlockListError",
    "DatasetNotFoundError",
    "DatasetWithScriptNotSupportedError",
    "DatasetWithTooComplexDataFilesPatternsError",
    "DatasetWithTooManyConfigsError",
    "DatasetWithTooManyParquetFilesError",
    "DatasetWithTooManySplitsError",
    "DiskError",
    "DuckDBIndexFileNotFoundError",
    "EmptyDatasetError",
    "ExternalServerError",
    "FeaturesError",
    "FeaturesResponseEmptyError",
    "FileFormatMismatchBetweenSplitsError",
    "FileSystemError",
    "HfHubError",
    "InfoError",
    "JobManagerCrashedError",
    "JobManagerExceededMaximumDurationError",
    "LockedDatasetTimeoutError",
    "MissingSpawningTokenError",
    "NoSupportedFeaturesError",
    "NotSupportedDisabledRepositoryError",
    "NotSupportedDisabledViewerError",
    "NotSupportedPrivateRepositoryError",
    "NotSupportedRepositoryNotFoundError",
    "NotSupportedTagNFAAError",
    "NormalRowsError",
    "ParameterMissingError",
    "ParquetResponseEmptyError",
    "PresidioScanNotEnabledForThisDataset",
    "PreviousStepFormatError",
    "PreviousStepStatusError",
    "PreviousStepStillProcessingError",
    "PolarsParquetReadError",
    "RetryableConfigNamesError",
    "RowsPostProcessingError",
    "SplitsNamesError",
    "SplitNamesFromStreamingError",
    "SplitNotFoundError",
    "SplitParquetSchemaMismatchError",
    "SplitWithTooBigParquetError",
    "StreamingRowsError",
    "TooBigContentError",
    "TooLongColumnNameError",
    "TooManyColumnsError",
    "UnexpectedError",
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


class DataFilesNotFoundError(CacheableError):
    """No (supported) data files found."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DataFilesNotFoundError", cause, False)


class DatasetGenerationError(CacheableError):
    """The dataset generation failed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetGenerationError", cause, True)


class DatasetGenerationCastError(CacheableError):
    """The dataset generation failed because of a cast error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetGenerationCastError", cause, True)


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


class DatasetWithTooManyConfigsError(CacheableError):
    """The number of configs of a dataset exceeded the limit."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManyConfigsError", cause, True)


class DatasetWithTooManySplitsError(CacheableError):
    """The number of splits of a dataset exceeded the limit."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManySplitsError", cause, True)


class DatasetWithTooManyParquetFilesError(CacheableError):
    """The number of parquet files of a dataset is too big."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithTooManyParquetFilesError", cause, True)


class DuckDBIndexFileNotFoundError(CacheableError):
    """No duckdb index file was found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "DuckDBIndexFileNotFoundError", cause, False)


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


class ExternalServerError(CacheableError):
    """The spawning.ai server is not responding."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ExternalServerError", cause, False)


class FeaturesError(CacheableError):
    """The features could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesError", cause, True)


class FeaturesResponseEmptyError(CacheableError):
    """No features were found in cache for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesResponseEmptyError", cause, True)


class FileFormatMismatchBetweenSplitsError(CacheableError):
    """Couldn't infer the same data file format for all splits."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message, HTTPStatus.INTERNAL_SERVER_ERROR, "FileFormatMismatchBetweenSplitsError", cause, False
        )


class FileSystemError(CacheableError):
    """An error happen reading from File System."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FileSystemError", cause, False)


class HfHubError(CacheableError):
    """The HF Hub server is not responding or gave an error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "HfHubError", cause, False)


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


class PolarsParquetReadError(CacheableError):
    """Error while reading parquet files with polars."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PolarsParquetReadError", cause, False)


class PreviousStepStillProcessingError(CacheableError):
    """The previous steps are still being processed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStillProcessingError", cause, False)


class RetryableConfigNamesError(CacheableError):
    """The config names could not be fetched, but we should retry."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "RetryableConfigNamesError", cause, True)


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


class SplitParquetSchemaMismatchError(CacheableError):
    """The Parquet files have different schemas, they should have identical column names."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="SplitParquetSchemaMismatchError",
            cause=cause,
            disclose_cause=False,
        )


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


class TooLongColumnNameError(CacheableError):
    """The column name is too long."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooLongColumnNameError", cause, True)


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


class DatasetWithScriptNotSupportedError(CacheableError):
    """We don't support some datasets because they have a dataset script."""

    def __init__(self, message: str = "", cause: Optional[BaseException] = None):
        message = message or (
            "The dataset viewer doesn't support this dataset because it runs arbitrary Python code. "
            "You can convert it to a Parquet data-only dataset by using the convert_to_parquet CLI from the datasets "
            "library. See: https://huggingface.co/docs/datasets/main/en/cli#convert-to-parquet"
        )  # TODO: Change URL after next datasets release
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetWithScriptNotSupportedError", cause, True)


class NotSupportedError(CacheableError):
    pass


class NotSupportedDisabledRepositoryError(NotSupportedError):
    """The repository is disabled."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "NotSupportedDisabledRepositoryError", cause, False)


class NotSupportedRepositoryNotFoundError(NotSupportedError):
    """The repository has not been found."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "NotSupportedRepositoryNotFoundError", cause, False)


class NotSupportedDisabledViewerError(NotSupportedError):
    """The dataset viewer is disabled in the dataset configuration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "NotSupportedDisabledViewerError", cause, False)


class NotSupportedPrivateRepositoryError(NotSupportedError):
    """The repository is private."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "NotSupportedPrivateRepositoryError", cause, False)


class NotSupportedTagNFAAError(NotSupportedError):
    """The dataset viewer is disabled because the dataset has the NFAA tag."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "NotSupportedTagNFAAError", cause, False)


class DatasetInBlockListError(NotSupportedError):
    """The dataset is in the list of blocked datasets."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "DatasetInBlockListError", cause, False)


class DatasetWithTooComplexDataFilesPatternsError(CacheableError):
    """We don't show code snippets for datasets with too complex data files patterns (that we didn't manage to simplify)."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message, HTTPStatus.INTERNAL_SERVER_ERROR, "DatasetWithTooComplexDataFilesPatternsError", cause, True
        )


class PresidioScanNotEnabledForThisDataset(CacheableError):
    """We've only enabled some datasets for presidio scans."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_IMPLEMENTED, "PresidioScanNotEnabledForThisDataset", cause, False)
