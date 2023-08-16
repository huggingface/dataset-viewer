# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Literal, Optional

from libcommon.exceptions import CustomError

ApiErrorCode = Literal[
    "AuthCheckHubRequestError",
    "ExternalAuthenticatedError",
    "ExternalUnauthenticatedError",
    "InvalidParameter",
    "JWKError",
    "JWTExpiredSignature",
    "JWTInvalidClaimRead",
    "JWTInvalidClaimSub",
    "JWTInvalidKeyOrAlgorithm",
    "JWTInvalidSignature",
    "JWTMissingRequiredClaim",
    "MissingProcessingStepsError",
    "MissingRequiredParameter",
    "ResponseNotFound",
    "ResponseNotReady",
    "TransformRowsProcessingError",
    "UnexpectedApiError",
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
    """The external authentication check failed or timed out."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message, HTTPStatus.INTERNAL_SERVER_ERROR, "AuthCheckHubRequestError", cause=cause, disclose_cause=False
        )


class ExternalAuthenticatedError(ApiError):
    """The external authentication check failed while the user was authenticated.

    Even if the external authentication server returns 403 in that case, we return 404 because
    we don't know if the dataset exist or not. It's also coherent with how the Hugging Face Hub works.

    TODO: should we return DatasetNotFoundError instead? maybe the error code is leaking existence of private datasets.
    """

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ExternalAuthenticatedError")


class ExternalUnauthenticatedError(ApiError):
    """The external authentication check failed while the user was unauthenticated."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "ExternalUnauthenticatedError")


class InvalidParameterError(ApiError):
    """A parameter has an invalid value."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNPROCESSABLE_ENTITY, "InvalidParameter")


class JWKError(ApiError):
    """The JWT key (JWK) could not be fetched or parsed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "JWKError", cause=cause, disclose_cause=False)


class MissingRequiredParameterError(ApiError):
    """A required parameter is missing."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.UNPROCESSABLE_ENTITY, "MissingRequiredParameter")


class ResponseNotFoundError(ApiError):
    """The response has not been found."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ResponseNotFound")


class ResponseNotReadyError(ApiError):
    """The response has not been processed yet."""

    def __init__(self, message: str):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ResponseNotReady")


class TransformRowsProcessingError(ApiError):
    """There was an error when transforming rows to list."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TransformRowsProcessingError", cause, True)


class JWTExpiredSignature(ApiError):
    """The JWT signature has expired."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "JWTExpiredSignature", cause, True)


class JWTInvalidClaimRead(ApiError):
    """The 'read' claim in the JWT payload is invalid."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "JWTInvalidClaimRead", cause, True)


class JWTInvalidClaimSub(ApiError):
    """The 'sub' claim in the JWT payload is invalid."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "JWTInvalidClaimSub", cause, True)


class JWTInvalidKeyOrAlgorithm(ApiError):
    """The key and the algorithm used to verify the JWT signature are not compatible."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "JWTInvalidKeyOrAlgorithm", cause, True)


class JWTInvalidSignature(ApiError):
    """The JWT signature verification failed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "JWTInvalidSignature", cause, True)


class JWTMissingRequiredClaim(ApiError):
    """A claim is missing in the JWT payload."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.UNAUTHORIZED, "JWTMissingRequiredClaim", cause, True)


class UnexpectedApiError(ApiError):
    """The server raised an unexpected error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        logging.error(message, exc_info=cause)
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "UnexpectedApiError", cause)
