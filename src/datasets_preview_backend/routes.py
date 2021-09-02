import logging
from typing import Union

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
    StatusError,
)
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.info import get_info
from datasets_preview_backend.queries.rows import extract_rows
from datasets_preview_backend.queries.splits import get_splits
from datasets_preview_backend.utils import get_int_value


def log_error(err: StatusError):
    logging.debug(
        f"Error {err.status_code} '{err.message}'. Caused by a {type(err.__cause__).__name__}: '{str(err.__cause__)}'"
    )


async def healthcheck(_: Request):
    return PlainTextResponse("ok")


async def info(request: Request):
    dataset: str = request.query_params.get("dataset")

    try:
        return JSONResponse(get_info(dataset))
    except (Status400Error, Status404Error) as err:
        log_error(err)
        return JSONResponse(err.as_dict(), status_code=err.status_code)
    # other exceptions will generate a 500 response


async def configs(request: Request):
    dataset: str = request.query_params.get("dataset")

    try:
        return JSONResponse(get_configs(dataset))
    except (Status400Error, Status404Error) as err:
        log_error(err)
        return JSONResponse(err.as_dict(), status_code=err.status_code)
    # other exceptions will generate a 500 response


async def splits(request: Request):
    dataset: str = request.query_params.get("dataset")
    config: Union[str, None] = request.query_params.get("config")

    try:
        return JSONResponse(get_splits(dataset, config))
    except (Status400Error, Status404Error) as err:
        log_error(err)
        return JSONResponse(err.as_dict(), status_code=err.status_code)
    # other exceptions will generate a 500 response


async def rows(request: Request):
    dataset: str = request.query_params.get("dataset")
    config: Union[str, None] = request.query_params.get("config")
    split: str = request.query_params.get("split")
    num_rows = get_int_value(d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT)

    try:
        return JSONResponse(extract_rows(dataset, config, split, num_rows))
    except (Status400Error, Status404Error) as err:
        log_error(err)
        return JSONResponse(err.as_dict(), status_code=err.status_code)
    # other exceptions will generate a 500 response
