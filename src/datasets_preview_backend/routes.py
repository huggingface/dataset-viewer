import logging
from typing import Optional, Union

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
from datasets_preview_backend.utils import get_int_value, get_token

logger = logging.getLogger(__name__)


def log_error(err: StatusError):
    logger.warning(f"Error {err.status_code} '{err.message}'.")
    logger.debug(f"Caused by a {type(err.__cause__).__name__}: '{str(err.__cause__)}'")


def get_response(
    route: str,
    dataset: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    num_rows: Optional[int] = None,
    use_auth_token: Optional[Union[str, None]] = None,
):
    try:
        if route == "/info":
            logger.info(f"/info, dataset={dataset}")
            content = get_info(dataset, use_auth_token)
        elif route == "/configs":
            logger.info(f"/configs, dataset={dataset}")
            content = get_configs(dataset, use_auth_token)
        elif route == "/splits":
            logger.info(f"/splits, dataset={dataset}, config={config}")
            content = get_splits(dataset, config, use_auth_token)
        elif route == "/rows":
            logger.info(f"/rows, dataset={dataset}, config={config}, split={split}, num_rows={num_rows}")
            content = extract_rows(dataset, config, split, num_rows, use_auth_token)
        else:
            raise Exception(f"unknown route {route}")
        return {"content": content, "status_code": 200}
    except (Status400Error, Status404Error) as err:
        log_error(err)
        return {"content": err.as_dict(), "status_code": err.status_code}


async def healthcheck(_: Request):
    logger.info("/healthcheck")
    return PlainTextResponse("ok")


async def info(request: Request):
    dataset = request.query_params.get("dataset")
    use_auth_token = get_token(request)

    response = get_response("/info", dataset=dataset, use_auth_token=use_auth_token)
    return JSONResponse(response["content"], status_code=response["status_code"])


async def configs(request: Request):
    dataset = request.query_params.get("dataset")
    use_auth_token = get_token(request)

    response = get_response("/configs", dataset=dataset, use_auth_token=use_auth_token)
    return JSONResponse(response["content"], status_code=response["status_code"])


async def splits(request: Request):
    dataset = request.query_params.get("dataset")
    config = request.query_params.get("config")
    use_auth_token = get_token(request)

    response = get_response("/splits", dataset=dataset, config=config, use_auth_token=use_auth_token)
    return JSONResponse(response["content"], status_code=response["status_code"])


async def rows(request: Request):
    dataset = request.query_params.get("dataset")
    config = request.query_params.get("config")
    split = request.query_params.get("split")
    num_rows = get_int_value(d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT)
    use_auth_token = get_token(request)

    response = get_response(
        "/rows", dataset=dataset, config=config, split=split, num_rows=num_rows, use_auth_token=use_auth_token
    )
    return JSONResponse(response["content"], status_code=response["status_code"])
