import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
import uvicorn

PORT = 8000


def healthcheck(request: Request):
    return PlainTextResponse("ok")


def start():
    app = Starlette(
        routes=[
            Route("/healthcheck", endpoint=healthcheck),
        ]
    )

    port = os.environ.get("TBL_PORT", PORT)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start()
