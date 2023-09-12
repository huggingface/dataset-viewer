# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import dataclasses
import json
import logging
from asyncio import CancelledError
from typing import AsyncGenerator, AsyncIterable

from libapi.utils import Endpoint
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import Request
from starlette.responses import Response

from sse_api.watcher import HubCacheWatcher

# https://sysid.github.io/server-sent-events/
# > Reliability: Maintaining an unique Id with messages the server can see that the client missed a number of messages
# > and send the backlog of missed messages on reconnect.
# ^ how do we manage errors and re-connections?

# Also: how do we manage multiple SSE servers / uvicorn workers / load-balancing?


def create_hub_cache_endpoint(hub_cache_watcher: HubCacheWatcher) -> Endpoint:
    async def hub_cache_endpoint(_: Request) -> Response:
        logging.info("/hub-cache")

        uuid, event = hub_cache_watcher.subscribe()
        init_task = hub_cache_watcher.run_initialization(uuid)

        async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
            try:
                while True:
                    new_value = await event.wait_value()
                    event.clear()
                    if new_value is not None:
                        yield ServerSentEvent(data=json.dumps(dataclasses.asdict(new_value)), event="message")
            finally:
                hub_cache_watcher.unsubscribe(uuid)
                await init_task

        return EventSourceResponse(error_handling(event_generator()), media_type="text/event-stream")

    return hub_cache_endpoint


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
