import asyncio
import socket
from typing import Optional

import uvicorn
from starlette.applications import Starlette
from uvicorn.config import Config


class UvicornServer(uvicorn.Server):
    """
    Wrapper around uvicorn.Server to be able to run it in async tests
    See https://github.com/encode/uvicorn/discussions/1103#discussioncomment-4240770
    """

    serve_task: asyncio.Task[Starlette]
    did_start: asyncio.Event
    did_close: asyncio.Event

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        self.did_start = asyncio.Event()
        self.did_close = asyncio.Event()

    async def start(self, sockets: Optional[list[socket.socket]] = None) -> None:
        self.serve_task = asyncio.create_task(self.serve(sockets=sockets))  # type: ignore
        self.serve_task.add_done_callback(lambda _: self.did_close.set())
        await self.did_start.wait()

    async def startup(self, sockets: Optional[list[socket.socket]] = None) -> None:
        await super().startup(sockets=sockets)  # type: ignore
        self.did_start.set()

    async def shutdown(self, sockets: Optional[list[socket.socket]] = None) -> None:
        await super().shutdown()
        self.serve_task.cancel()
        await self.did_close.wait()
