# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from asyncio import CancelledError
from typing import AsyncGenerator, AsyncIterable

from libapi.utils import Endpoint
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import Request
from starlette.responses import Response

from sse_api.watcher import RandomValueWatcher

# https://sysid.github.io/server-sent-events/
# > Reliability: Maintaining an unique Id with messages the server can see that the client missed a number of messages
# > and send the backlog of missed messages on reconnect.
# ^ how do we manage errors and re-connections?

# Also: how do we manage multiple SSE servers / uvicorn workers / load-balancing?


def create_numbers_endpoint(random_value_watcher: RandomValueWatcher) -> Endpoint:
    async def numbers_endpoint(_: Request) -> Response:
        logging.info("/numbers")

        uuid, event = random_value_watcher.subscribe()

        async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
            previous_value = None
            try:
                while True:
                    new_value = await event.wait_random_value()
                    event.clear()

                    if new_value == previous_value:
                        continue
                    yield ServerSentEvent(data=new_value, event="message")
            finally:
                random_value_watcher.unsubscribe(uuid)

        return EventSourceResponse(error_handling(event_generator()), media_type="text/event-stream")

    return numbers_endpoint


async def error_handling(
    sse_generator: AsyncGenerator[ServerSentEvent, None],
) -> AsyncIterable[ServerSentEvent]:
    try:
        async for event in sse_generator:
            yield event
    except CancelledError:
        yield ServerSentEvent("Connection closed", event="error")
        raise
    except Exception:
        yield ServerSentEvent("Internal server error", event="error")
        raise
