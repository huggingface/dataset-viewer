# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Literal, Optional

import requests
from libcommon.prometheus import StepProfiler
from requests import PreparedRequest
from requests.auth import AuthBase
from starlette.requests import Request

from api.jwt_token import is_jwt_valid
from api.utils import (
    AuthCheckHubRequestError,
    ExternalAuthenticatedError,
    ExternalUnauthenticatedError,
)


class RequestAuth(AuthBase):
    """Attaches input Request authentication headers to the given Request object."""

    def __init__(self, request: Optional[Request]) -> None:
        if request is not None:
            self.cookie = request.headers.get("cookie")
            self.authorization = request.headers.get("authorization")
        else:
            self.cookie = None
            self.authorization = None

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        # modify and return the request
        if self.cookie:
            r.headers["cookie"] = self.cookie
        if self.authorization:
            r.headers["authorization"] = self.authorization
        return r


def auth_check(
    dataset: str,
    external_auth_url: Optional[str] = None,
    request: Optional[Request] = None,
    hf_jwt_public_key: Optional[str] = None,
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
        hf_jwt_public_key (str|None): the public key to use to decode the JWT token
        hf_jwt_algorithm (str): the algorithm to use to decode the JWT token
        hf_timeout_seconds (float|None): the timeout in seconds for the external authentication service. It
          is used both for the connection timeout and the read timeout. If None, the request never timeouts.

    Returns:
        None: the dataset is authorized for the request
    """
    with StepProfiler(method="auth_check", step="all"):
        with StepProfiler(method="auth_check", step="prepare parameters"):
            if request:
                logging.debug(
                    f"Looking if jwt is passed in request headers {request.headers} to bypass authentication."
                )
                HEADER = "x-api-key"
                if token := request.headers.get(HEADER):
                    if is_jwt_valid(
                        dataset=dataset, token=token, public_key=hf_jwt_public_key, algorithm=hf_jwt_algorithm
                    ):
                        logging.debug(
                            f"By-passing the authentication step, because a valid JWT was passed in header: '{HEADER}'"
                            f" for dataset {dataset}. JWT was: {token}"
                        )
                        return True
                    logging.debug(
                        f"Error while validating the JWT passed in header: '{HEADER}' for dataset {dataset}. Trying"
                        f" with the following authentication mechanisms. JWT was: {token}"
                    )
                else:
                    logging.debug(
                        f"No JWT was passed in header: '{HEADER}' for dataset {dataset}. Trying with the following"
                        " authentication mechanisms."
                    )
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
                response = requests.get(url, auth=auth, timeout=hf_timeout_seconds)
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
        elif response.status_code in [403, 404]:
            raise ExternalAuthenticatedError(
                "The dataset does not exist, or is not accessible with the current credentials (private or gated)."
                " Please check the spelling of the dataset name or retry with other authentication credentials."
            )
        else:
            raise ValueError(f"Unexpected status code {response.status_code}")
