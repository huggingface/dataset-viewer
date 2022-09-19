# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

logger = logging.getLogger(__name__)


async def healthcheck_endpoint(_: Request) -> Response:
    logger.info("/healthcheck")
    return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})
