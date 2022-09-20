# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Mapping, Tuple, Union

from requests import PreparedRequest
from responses import _Body


def request_callback(request: PreparedRequest) -> Union[Exception, Tuple[int, Mapping[str, str], _Body]]:
    # return 401 if a cookie has been provided, 404 if a token has been provided,
    # and 401 if none has been provided
    # there is no logic behind this behavior, it's just to test if the cookie and
    # token are correctly passed to the auth_check service
    if request.headers.get("cookie"):
        return (401, {"Content-Type": "text/plain"}, "OK")
    if request.headers.get("authorization"):
        return (404, {"Content-Type": "text/plain"}, "OK")
    return (200, {"Content-Type": "text/plain"}, "OK")
