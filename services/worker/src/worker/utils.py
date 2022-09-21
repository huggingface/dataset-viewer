# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import functools
import time
from http import HTTPStatus
from logging import Logger
from typing import Literal, Optional

from libutils.exceptions import CustomError

WorkerErrorCode = Literal[
    "DatasetNotFoundError",
    "ConfigNotFoundError",
    "SplitNotFoundError",
    "SplitsNamesError",
    "EmptyDatasetError",
    "InfoError",
    "FeaturesError",
    "StreamingRowsError",
    "NormalRowsError",
    "RowsPostProcessingError",
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


class ConfigNotFoundError(WorkerCustomError):
    """Raised when the config does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ConfigNotFoundError", cause, False)


class SplitNotFoundError(WorkerCustomError):
    """Raised when the split does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "SplitNotFoundError", cause, False)


class SplitsNamesError(WorkerCustomError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class EmptyDatasetError(WorkerCustomError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class InfoError(WorkerCustomError):
    """Raised when the info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class FeaturesError(WorkerCustomError):
    """Raised when the features could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FeaturesError", cause, True)


class StreamingRowsError(WorkerCustomError):
    """Raised when the rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class NormalRowsError(WorkerCustomError):
    """Raised when the rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)


class RowsPostProcessingError(WorkerCustomError):
    """Raised when the rows could not be post-processed successfully."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "RowsPostProcessingError", cause, False)


class UnexpectedError(WorkerCustomError):
    """Raised when the response for the split has not been found."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "UnexpectedError", cause, False)


def retry(logger: Logger):
    def decorator_retry(func):
        """retries with an increasing sleep before every attempt"""
        SLEEPS = [1, 7, 70, 7 * 60, 70 * 60]
        MAX_ATTEMPTS = len(SLEEPS)

        @functools.wraps(func)
        def decorator(*args, **kwargs):
            attempt = 0
            last_err = None
            while attempt < MAX_ATTEMPTS:
                try:
                    """always sleep before calling the function. It will prevent rate limiting in the first place"""
                    duration = SLEEPS[attempt]
                    logger.info(f"Sleep during {duration} seconds to preventively mitigate rate limiting.")
                    time.sleep(duration)
                    return func(*args, **kwargs)
                except ConnectionError as err:
                    logger.info("Got a ConnectionError, possibly due to rate limiting. Let's retry.")
                    last_err = err
                    attempt += 1
            raise RuntimeError(f"Give up after {attempt} attempts with ConnectionError") from last_err

        return decorator

    return decorator_retry
