from typing import Literal, Optional

import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from starlette.requests import Request

from api.utils import ExternalAuthenticatedError, ExternalUnauthenticatedError


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


def auth_check(dataset: str, external_auth_url: str = "", request: Optional[Request] = None) -> Literal[True]:
    """check if the dataset is authorized for the request

    Args:
        dataset (str): the dataset name
        external_auth_url (str): the URL of an external authentication service. The URL must contain `%s`,
          which will be replaced with the dataset name, for example: https://huggingface.co/api/datasets/%s/auth-check
          The authentication service must follow the specification in
          https://nginx.org/en/docs/http/ngx_http_auth_request_module.html and return 200, 401 or 403.
          If empty, the dataset is always authorized.
        request (Request | None): the request which optionally bears authentication headers: "cookie" or
          "authorization"

    Returns:
        None: the dataset is authorized for the request
    """
    if not external_auth_url:
        return True
    try:
        url = external_auth_url % dataset
    except TypeError as e:
        raise ValueError("external_auth_url must contain %s") from e
    try:
        response = requests.get(url, auth=RequestAuth(request))
    except Exception as err:
        raise RuntimeError("External authentication check failed", err) from err
    if response.status_code == 200:
        return True
    elif response.status_code == 401:
        raise ExternalUnauthenticatedError(
            "The dataset is not authorized for the request. Please retry with authentication."
        )

    elif response.status_code == 403:
        raise ExternalAuthenticatedError(
            "The dataset is not authorized for the request with the provided authentication."
        )

    else:
        raise ValueError(f"Unexpected status code {response.status_code}")
