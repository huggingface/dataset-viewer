import json
import logging
from typing import Optional, Union

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT, cache
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

logger = logging.getLogger(__name__)


def log_error(err: StatusError):
    logger.warning(f"Error {err.status_code} '{err.message}'.")
    logger.debug(f"Caused by a {type(err.__cause__).__name__}: '{str(err.__cause__)}'")


def get_response(
    *,
    route: str,
    dataset: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    num_rows: Optional[int] = None,
    token: Optional[Union[str, None]] = None,
):
    try:
        if route == "/info":
            logger.info(f"/info, dataset={dataset}")
            content = get_info(dataset, token)
        elif route == "/configs":
            logger.info(f"/configs, dataset={dataset}")
            content = get_configs(dataset, token)
        elif route == "/splits":
            logger.info(f"/splits, dataset={dataset}, config={config}")
            content = get_splits(dataset, config, token)
        elif route == "/rows":
            logger.info(f"/rows, dataset={dataset}, config={config}, split={split}, num_rows={num_rows}")
            content = extract_rows(dataset, config, split, num_rows, token)
        else:
            raise Exception(f"unknown route {route}")
        status_code = 200
    except (Status400Error, Status404Error) as err:
        log_error(err)
        content = err.as_dict()
        status_code = err.status_code

    # response content is encoded to avoid issues when caching ("/info" returns a non pickable object)
    bytes_content = json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "content": bytes_content,
        "status_code": status_code,
    }


def get_cached_response(**kwargs):
    cache.close()
    key = kwargs
    print(kwargs)
    if key in cache:
        response = cache.get(key)
    else:
        response = get_response(**kwargs)
        cache.set(key, response)
    return response


class CustomJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: bytes) -> bytes:
        # content is already an UTF-8 encoded JSON
        return content


class HealthCheck(HTTPEndpoint):
    async def get(self, request: Request):
        logger.info("/healthcheck")
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})


class Info(HTTPEndpoint):
    async def get(self, request: Request):
        dataset = request.query_params.get("dataset")

        response = get_cached_response(route="/info", dataset=dataset, token=request.user.token)
        return CustomJSONResponse(response["content"], status_code=response["status_code"])


class Configs(HTTPEndpoint):
    async def get(self, request: Request):
        dataset = request.query_params.get("dataset")

        response = get_cached_response(route="/configs", dataset=dataset, token=request.user.token)
        return CustomJSONResponse(response["content"], status_code=response["status_code"])


class Splits(HTTPEndpoint):
    async def get(self, request: Request):
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")

        response = get_cached_response(route="/splits", dataset=dataset, config=config, token=request.user.token)
        return CustomJSONResponse(response["content"], status_code=response["status_code"])


class Rows(HTTPEndpoint):
    async def get(self, request: Request):
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        split = request.query_params.get("split")
        num_rows = get_int_value(d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT)

        response = get_cached_response(
            route="/rows", dataset=dataset, config=config, split=split, num_rows=num_rows, token=request.user.token
        )
        return CustomJSONResponse(response["content"], status_code=response["status_code"])
