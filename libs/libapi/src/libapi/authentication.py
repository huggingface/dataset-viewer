# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from collections.abc import Generator
from typing import Literal, Optional

import httpx
from libcommon.prometheus import StepProfiler
from starlette.requests import Request

from libapi.exceptions import (
    AuthCheckHubRequestError,
    ExternalAuthenticatedError,
    ExternalUnauthenticatedError,
)
from libapi.jwt_token import validate_jwt


class RequestAuth(httpx.Auth):
    """Attaches input Request authentication headers to the given Request object."""

    def __init__(self, request: Optional[Request]) -> None:
        self.cookie = request.headers.get("cookie") if request else None
        self.authorization = request.headers.get("authorization") if request else None

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        # modify and yield the request
        if self.cookie:
            request.headers["cookie"] = self.cookie
        if self.authorization:
            request.headers["authorization"] = self.authorization
        yield request


def get_jwt_token(request: Optional[Request] = None) -> Optional[str]:
    if not request:
        return None
    # x-api-token is deprecated and will be removed in the future
    if token := request.headers.get("x-api-key"):
        return token
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    token = authorization.removeprefix("Bearer jwt:")
    return None if token == authorization else token


async def auth_check(
    dataset: str,
    external_auth_url: Optional[str] = None,
    request: Optional[Request] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Literal[True]:
    """check if the dataset is authorized for the request

    It sends a request to the Hugging Face API to check if the dataset is authorized for the input request. The request
    to the Hugging Face API is authenticated with the same authentication headers as the input request. It timeouts
    after 200ms.

    Args:
        dataset (str): the dataset name
        external_auth_url (str|None): the URL of an external authentication service. The URL must contain `%s`,
          which will be replaced with the dataset name, for example: https://huggingface.co/api/datasets/%s/auth-check
          The authentication service must return 200, 401, 403 or 404.
          If None, the dataset is always authorized.
        request (Request | None): the request which optionally bears authentication headers: "cookie",
          "authorization" or "X-Api-Key"
        hf_jwt_public_keys (list[str]|None): the public keys to use to decode the JWT token
        hf_jwt_algorithm (str): the algorithm to use to decode the JWT token
        hf_timeout_seconds (float|None): the timeout in seconds for the external authentication service. It
          is used both for the connection timeout and the read timeout. If None, the request never timeouts.

    Returns:
        None: the dataset is authorized for the request
    """
    with StepProfiler(method="auth_check", step="all"):
        with StepProfiler(method="auth_check", step="check JWT"):
            if (jwt_token := get_jwt_token(request)) and hf_jwt_public_keys and hf_jwt_algorithm:
                validate_jwt(
                    dataset=dataset, token=jwt_token, public_keys=hf_jwt_public_keys, algorithm=hf_jwt_algorithm
                )
                logging.debug(
                    "By-passing the authentication step, because a valid JWT was passed in headers"
                    f" for dataset {dataset}. JWT was: {jwt_token}"
                )
                return True
        with StepProfiler(method="auth_check", step="prepare parameters"):
            if external_auth_url is None:
                return True
            try:
                url = external_auth_url % dataset
            except TypeError as e:
                raise ValueError("external_auth_url must contain %s") from e
        with StepProfiler(method="auth_check", step="create auth parameter"):
            auth = RequestAuth(request)
        with StepProfiler(
            method="auth_check",
            step="requests.get",
            context=f"external_auth_url={external_auth_url} timeout={hf_timeout_seconds}",
        ):
            try:
                logging.debug(
                    f"Checking authentication on the Hugging Face Hub for dataset {dataset}, url: {url}, timeout:"
                    f" {hf_timeout_seconds}, authorization: {auth.authorization}"
                )
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, auth=auth, timeout=hf_timeout_seconds)
            except Exception as err:
                raise AuthCheckHubRequestError(
                    (
                        "Authentication check on the Hugging Face Hub failed or timed out. Please try again later,"
                        " it's a temporary internal issue."
                    ),
                    err,
                ) from err
    with StepProfiler(method="auth_check", step="return or raise"):
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            raise ExternalUnauthenticatedError(
                "The dataset does not exist, or is not accessible without authentication (private or gated). Please"
                " check the spelling of the dataset name or retry with authentication."
            )
        elif response.status_code in {403, 404}:
            raise ExternalAuthenticatedError(
                "The dataset does not exist, or is not accessible with the current credentials (private or gated)."
                " Please check the spelling of the dataset name or retry with other authentication credentials."
            )
        else:
            raise ValueError(f"Unexpected status code {response.status_code}")
