import logging

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse, Response

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.configs import get_configs_response
from datasets_preview_backend.queries.datasets import get_datasets_response
from datasets_preview_backend.queries.info import get_info_response
from datasets_preview_backend.queries.rows import get_rows_response
from datasets_preview_backend.queries.splits import get_splits_response
from datasets_preview_backend.responses import send
from datasets_preview_backend.utils import get_int_value

logger = logging.getLogger(__name__)


class HealthCheck(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/healthcheck")
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})


class CacheStats(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache")
        return JSONResponse(get_cache_stats(), headers={"Cache-Control": "no-store"})


class Datasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/datasets")
        response, max_age = get_datasets_response(_return_max_age=True)
        return send(response, max_age)


class Info(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/info, dataset={dataset}")
        response, max_age = get_info_response(dataset=dataset, token=request.user.token, _return_max_age=True)
        return send(response, max_age)


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")
        response, max_age = get_configs_response(dataset=dataset, token=request.user.token, _return_max_age=True)
        return send(response, max_age)


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")
        response, max_age = get_splits_response(
            dataset=dataset, config=config, token=request.user.token, _return_max_age=True
        )
        return send(response, max_age)


class Rows(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        split = request.query_params.get("split")
        num_rows = get_int_value(d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT)
        logger.info(f"/rows, dataset={dataset}, config={config}, split={split}, num_rows={num_rows}")
        response, max_age = get_rows_response(
            dataset=dataset,
            config=config,
            split=split,
            num_rows=num_rows,
            token=request.user.token,
            _return_max_age=True,
        )
        return send(response, max_age)
