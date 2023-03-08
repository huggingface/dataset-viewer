# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from api.prometheus import StepProfiler


async def healthcheck_endpoint(_: Request) -> Response:
    logging.info("/healthcheck")
    with StepProfiler(method="healthcheck_endpoint", step="all"):
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})
