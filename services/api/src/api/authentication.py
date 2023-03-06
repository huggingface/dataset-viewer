# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Literal, Optional

import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from starlette.requests import Request

from api.prometheus import StepProfiler
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
    external_auth_bypass_key: Optional[str] = None,
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
        request (Request | None): the request which optionally bears authentication headers: "cookie" or
          "authorization"
        external_auth_bypass_key (str|None): if the request bears this secret key in the "X-Api-Key" header, the
          external authentication service is bypassed.
        hf_timeout_seconds (float|None): the timeout in seconds for the external authentication service. It
          is used both for the connection timeout and the read timeout. If None, the request never timeouts.

    Returns:
        None: the dataset is authorized for the request
    """
    with StepProfiler(method="auth_check", step="all"):
        with StepProfiler(method="auth_check", step="prepare parameters"):
            if (
                external_auth_bypass_key is not None
                and request is not None
                and request.headers.get("X-Api-Key") == external_auth_bypass_key
            ):
                return True
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
                    f" {hf_timeout_seconds}"
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
