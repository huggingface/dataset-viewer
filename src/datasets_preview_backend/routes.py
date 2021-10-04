import logging

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.responses import get_endpoint_response

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
        return get_endpoint_response(endpoint="/datasets").send()


class Infos(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/infos, dataset={dataset}")
        return get_endpoint_response(endpoint="/infos", dataset=dataset, config=config).send()


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")
        return get_endpoint_response(endpoint="/configs", dataset=dataset).send()


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")
        return get_endpoint_response(
            endpoint="/splits",
            dataset=dataset,
            config=config,
        ).send()


class Rows(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        split = request.query_params.get("split")
        logger.info(f"/rows, dataset={dataset}, config={config}, split={split}")
        return get_endpoint_response(
            endpoint="/rows",
            dataset=dataset,
            config=config,
            split=split,
        ).send()
