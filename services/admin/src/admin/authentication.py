# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Literal, Optional

import httpx
from libapi.authentication import RequestAuth
from libapi.exceptions import ExternalAuthenticatedError, ExternalUnauthenticatedError
from starlette.requests import Request


async def auth_check(
    external_auth_url: Optional[str] = None,
    request: Optional[Request] = None,
    organization: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Literal[True]:
    """check if the user is member of the organization

    Args:
        external_auth_url (str | None): the URL of an external authentication service. If None, the dataset is always
          authorized.
        request (Request | None): the request which optionally bears authentication headers: "cookie" or
          "authorization"
        organization (str | None): the organization name. If None, the dataset is always
          authorized.
        hf_timeout_seconds (float | None): the timeout in seconds for the HTTP request to the external authentication
          service.

    Returns:
        None: the user is authorized
    """
    if organization is None or external_auth_url is None:
        return True
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(external_auth_url, auth=RequestAuth(request), timeout=hf_timeout_seconds)
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
    elif response.status_code in {403, 404}:
        raise ExternalAuthenticatedError(
            "Cannot access the route with the current credentials. Please retry with other authentication credentials."
        )
    else:
        raise ValueError(f"Unexpected status code {response.status_code}")
