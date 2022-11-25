# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Literal, Optional

from libcommon.exceptions import CustomError

WorkerErrorCode = Literal[
    "DatasetNotFoundError",
    "EmptyDatasetError",
    "SplitsNamesError",
    "UnexpectedError",
]


class WorkerCustomError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: WorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class DatasetNotFoundError(WorkerCustomError):
    """Raised when the dataset does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "DatasetNotFoundError", cause, False)


class SplitsNamesError(WorkerCustomError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class EmptyDatasetError(WorkerCustomError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class UnexpectedError(WorkerCustomError):
    """Raised when the response for the split has not been found."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "UnexpectedError", cause, False)
