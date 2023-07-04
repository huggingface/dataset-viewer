# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Literal, Optional

from libcommon.exceptions import CustomError

ApiErrorCode = Literal["JWKError"]


class ApiError(CustomError):
    """Base class for exceptions raised by an API service."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ApiErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class JWKError(ApiError):
    """Raised when the JWT key (JWK) could not be fetched or parsed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "JWKError", cause=cause, disclose_cause=False)
