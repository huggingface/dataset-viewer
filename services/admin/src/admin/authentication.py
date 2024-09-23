# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Sequence
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
    require_fine_grained_permissions: Sequence[str] = ("repo.write",),
) -> Literal[True]:
    """check if the user is member of the organization

    Args:
        external_auth_url (`str`, *optional*): the URL of an external authentication service. If None, the dataset is always
          authorized.
        request (`Request`, *optional*): the request which optionally bears authentication header: "authorization"
        organization (`str`, *optional*): the organization name. If None, the dataset is always
          authorized.
        hf_timeout_seconds (`float`, *optional*): the timeout in seconds for the HTTP request to the external authentication
          service.
        require_fine_grained_permissions (`Sequence[str]`): require a fine-grained token with certain permissions
          for the organization, if organization is provided. Defaults to ("repo.write",).

    Returns:
        `Literal[True]`: the user is authorized
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
            if organization is None or (
                organization in {org["name"] for org in json["orgs"]}
                and json["auth"]["type"] == "access_token"
                and "fineGrained" in json["auth"]["accessToken"]
                and any(
                    set(permission for permission in scope["permissions"]) >= set(require_fine_grained_permissions)
                    for scope in json["auth"]["accessToken"]["fineGrained"]["scoped"]
                    if scope["entity"]["name"] == organization and scope["entity"]["type"] == "org"
                )
            ):
                return True
            else:
                raise ExternalAuthenticatedError(
                    "Cannot access the route with the current credentials. Please retry with other authentication credentials."
                )
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
