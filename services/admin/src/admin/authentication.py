# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Literal, Optional

import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from starlette.requests import Request

from admin.utils import ExternalAuthenticatedError, ExternalUnauthenticatedError


class RequestAuth(AuthBase):
    """Attaches input Request authentication headers to the given Request object."""

    def __init__(self, request: Optional[Request]) -> None:
        if request is not None:
            self.authorization = request.headers.get("authorization")
        else:
            self.authorization = None

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        # modify and return the request
        if self.authorization:
            r.headers["authorization"] = self.authorization
        return r


def auth_check(
    external_auth_url: Optional[str] = None, request: Optional[Request] = None, organization: Optional[str] = None
) -> Literal[True]:
    """check if the user is member of the organization

    Args:
        external_auth_url (str | None): the URL of an external authentication service. If None, the dataset is always
        authorized.
        request (Request | None): the request which optionally bears authentication headers: "cookie" or
          "authorization"
        organization (str | None): the organization name. If None, the dataset is always
        authorized.

    Returns:
        None: the user is authorized
    """
    if organization is None or external_auth_url is None:
        return True
    try:
        response = requests.get(external_auth_url, auth=RequestAuth(request))
    except Exception as err:
        raise RuntimeError("External authentication check failed", err) from err
    if response.status_code == 200:
        try:
            json = response.json()
            if organization is None or organization in {org["name"] for org in json["orgs"]}:
                return True
            else:
                raise ExternalAuthenticatedError("You are not member of the organization")
        except Exception as err:
            raise ExternalAuthenticatedError(
                "Cannot access the route with the current credentials. Please retry with other authentication"
                " credentials."
            ) from err
    elif response.status_code == 401:
        raise ExternalUnauthenticatedError("Cannot access the route. Please retry with authentication.")
    elif response.status_code in [403, 404]:
        raise ExternalAuthenticatedError(
            "Cannot access the route with the current credentials. Please retry with other authentication credentials."
        )
    else:
        raise ValueError(f"Unexpected status code {response.status_code}")
