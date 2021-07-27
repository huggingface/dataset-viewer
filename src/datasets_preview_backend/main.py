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
    print(f"Asked for {num_rows} first rows of model {model_id}")
    try:
        dataset = load_dataset(model_id, split="train", streaming=True)
    except:
        print(f"Dataset could not be loaded.")
        return []

    print(f"Dataset loaded")

    rows = list(dataset.take(num_rows))

    if len(rows) != num_rows:
        print(f"WARN could not read all the required rows ({len(rows)} / {num_rows})")

    return rows


async def extract(request: Request):
    model_id: str = request.path_params["model_id"]
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    return JSONResponse(get_dataset_extract(model_id, num_rows))


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
