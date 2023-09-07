# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
import logging
from typing import AsyncGenerator, TypedDict

from sse_starlette import EventSourceResponse
from starlette.requests import Request
from starlette.responses import Response

# https://sysid.github.io/server-sent-events/
# > Reliability: Maintaining an unique Id with messages the server can see that the client missed a number of messages
# > and send the backlog of missed messages on reconnect.
# ^ how do we manage errors and re-connections?

# Also: how do we manage multiple SSE servers / uvicorn workers / load-balancing?


class IntData(TypedDict):
    data: int


async def numbers(minimum: int, maximum: int) -> AsyncGenerator[IntData, None]:
    for i in range(minimum, maximum + 1):
        await asyncio.sleep(0.05)
        yield dict(data=i)


async def numbers_endpoint(_: Request) -> Response:
    logging.info("/numbers")
    generator = numbers(1, 5)
    return EventSourceResponse(generator)
