import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
import uvicorn

DEFAULT_PORT = 8000
DEFAULT_EXTRACT_ROWS_LIMIT = 100


def get_int_value(d, key, default):
    try:
        value = int(d.get(key))
    except TypeError:
        value = default
    return value


PORT = get_int_value(d=os.environ, key="DPB_PORT", default=DEFAULT_PORT)
EXTRACT_ROWS_LIMIT = get_int_value(
    d=os.environ, key="DPB_EXTRACT_ROWS_LIMIT", default=DEFAULT_EXTRACT_ROWS_LIMIT
)


async def healthcheck(request: Request):
    return PlainTextResponse("ok")


async def extract(request: Request):
    model_id: str = request.path_params["model_id"]
    rows = get_int_value(d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT)

    return PlainTextResponse(model_id + "-" + str(rows))


def start():
    app = Starlette(
        routes=[
            Route("/healthcheck", endpoint=healthcheck),
            Route("/{model_id:path}/extract", endpoint=extract),
        ]
    )

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    start()
