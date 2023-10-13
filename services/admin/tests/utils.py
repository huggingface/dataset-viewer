# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import httpx


def request_callback(request: httpx.Request) -> httpx.Response:
    # return 404 if a token has been provided,
    # and 200 if none has been provided
    # there is no logic behind this behavior, it's just to test if the
    # tokens are correctly passed to the auth_check service
    body = '{"orgs": [{"name": "org1"}]}'
    if request.headers.get("authorization"):
        return httpx.Response(status_code=404, text=body)
    return httpx.Response(status_code=200, text=body)
