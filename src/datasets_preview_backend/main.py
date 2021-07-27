import logging
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse
from starlette.routing import Route
import uvicorn

from datasets import load_dataset

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


def get_dataset_extract(model_id: str, num_rows: int):
    # TODO: manage splits and submodels
    logging.debug(f"Asked for {num_rows} first rows of model {model_id}")

    dataset = load_dataset(model_id, split="train", streaming=True)

    logging.debug(f"Dataset loaded")

    rows = list(dataset.take(num_rows))

    if len(rows) != num_rows:
        logging.warning(
            f"could not read all the required rows ({len(rows)} / {num_rows})"
        )

    return rows


async def extract(request: Request):
    model_id: str = request.path_params["model_id"]
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    try:
        return JSONResponse(get_dataset_extract(model_id, num_rows))
    except FileNotFoundError as e:
        return PlainTextResponse("Model data could not be found", status_code=404)


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
