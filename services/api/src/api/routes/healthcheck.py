# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.prometheus import StepProfiler
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response


async def healthcheck_endpoint(_: Request) -> Response:
    logging.info("/healthcheck")
    with StepProfiler(method="healthcheck_endpoint", step="all"):
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})
