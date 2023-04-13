# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from io import BufferedReader
from typing import Mapping, Tuple, Union

from requests import PreparedRequest
from responses import Response

_Body = Union[str, BaseException, Response, BufferedReader, bytes]


def request_callback(request: PreparedRequest) -> Union[Exception, Tuple[int, Mapping[str, str], _Body]]:
    # return 404 if a token has been provided,
    # and 200 if none has been provided
    # there is no logic behind this behavior, it's just to test if th
    # token are correctly passed to the auth_check service
    body = '{"orgs": [{"name": "org1"}]}'
    if request.headers.get("authorization"):
        return (404, {"Content-Type": "text/plain"}, body)
    return (200, {"Content-Type": "text/plain"}, body)
