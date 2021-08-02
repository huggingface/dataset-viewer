from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.splits import get_splits
from datasets_preview_backend.queries.rows import extract_rows
from datasets_preview_backend.utils import get_int_value
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
)


async def healthcheck(_: Request):
    return PlainTextResponse("ok")


async def configs(request: Request):
    dataset: str = request.query_params.get("dataset")

    if dataset is None:
        return PlainTextResponse(
            "'dataset' is a required query parameter.", status_code=400
        )

    try:
        return JSONResponse(get_configs(dataset))
    except (Status400Error, Status404Error) as err:
        return PlainTextResponse(err.message, status_code=err.status_code)
    # other exceptions will generate a 500 response


async def splits(request: Request):
    dataset: str = request.query_params.get("dataset")
    config: str = request.query_params.get("config")

    if dataset is None:
        return PlainTextResponse(
            "'dataset' is a required query parameter.", status_code=400
        )
    # note: config_name must not be set to refer to the None config_name (the majority of datasets).

    try:
        return JSONResponse(get_splits(dataset, config))
    except (Status400Error, Status404Error) as err:
        return PlainTextResponse(err.message, status_code=err.status_code)
    # other exceptions will generate a 500 response


async def rows(request: Request):
    dataset: str = request.query_params.get("dataset")
    config: str = request.query_params.get("config")
    split: str = request.query_params.get("split")
    num_rows = get_int_value(
        d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT
    )

    if dataset is None:
        return PlainTextResponse(
            "'dataset' is a required query parameter.", status_code=400
        )
    # note: config_name must not be set to refer to the None config_name (the majority of datasets).
    if split is None:
        return PlainTextResponse(
            "'split' is a required query parameter.", status_code=400
        )

    try:
        return JSONResponse(extract_rows(dataset, config, split, num_rows))
    except (Status400Error, Status404Error) as err:
        return PlainTextResponse(err.message, status_code=err.status_code)
    # other exceptions will generate a 500 response
