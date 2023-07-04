# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Literal, Optional

from libcommon.exceptions import CustomError

ApiErrorCode = Literal[
    "AuthCheckHubRequestError", "ExternalAuthenticatedError", "ExternalUnauthenticatedError", "JWKError"
]


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


class AuthCheckHubRequestError(ApiError):
    """Raised when the external authentication check failed or timed out."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message, HTTPStatus.INTERNAL_SERVER_ERROR, "AuthCheckHubRequestError", cause=cause, disclose_cause=False
        )


class ExternalAuthenticatedError(ApiError):
    """Raised when the external authentication check failed while the user was authenticated.

    Even if the external authentication server returns 403 in that case, we return 404 because
    we don't know if the dataset exist or not. It's also coherent with how the Hugging Face Hub works."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ExternalAuthenticatedError")


class ExternalUnauthenticatedError(ApiError):
    """Raised when the external authentication check failed while the user was unauthenticated."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "ExternalUnauthenticatedError")


class JWKError(ApiError):
    """Raised when the JWT key (JWK) could not be fetched or parsed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "JWKError", cause=cause, disclose_cause=False)
