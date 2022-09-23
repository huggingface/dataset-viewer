# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from werkzeug.wrappers.request import Request
from werkzeug.wrappers.response import Response


def auth_callback(request: Request) -> Response:
    # return 401 if a cookie has been provided, 404 if a token has been provided,
    # and 200 if none has been provided
    #
    # caveat: the returned status codes don't simulate the reality
    # they're just used to check every case
    return Response(
        status=401 if request.headers.get("cookie") else 404 if request.headers.get("authorization") else 200
    )
