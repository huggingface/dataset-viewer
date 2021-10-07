import logging

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from datasets_preview_backend.cache_entries import memoized_functions
from datasets_preview_backend.queries.cache_reports import get_cache_reports
from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.validity_status import get_valid_datasets
from datasets_preview_backend.responses import get_cached_response

logger = logging.getLogger(__name__)


technical_functions = {
    "/cache": get_cache_stats,
    "/cache-reports": get_cache_reports,
    "/valid": get_valid_datasets,
}


class HealthCheck(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/healthcheck")
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})


class CacheReports(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache-reports")
        return get_cached_response(memoized_functions=technical_functions, endpoint="/valid-reports").send()


class CacheStats(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache")
        return get_cached_response(memoized_functions=technical_functions, endpoint="/cache").send()


class ValidDatasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/valid")
        return get_cached_response(memoized_functions=technical_functions, endpoint="/valid").send()


class Datasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/datasets")
        return get_cached_response(memoized_functions=memoized_functions, endpoint="/datasets").send()


class Infos(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/infos, dataset={dataset}")
        return get_cached_response(
            memoized_functions=memoized_functions, endpoint="/infos", dataset=dataset, config=config
        ).send()


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")
        return get_cached_response(memoized_functions=memoized_functions, endpoint="/configs", dataset=dataset).send()


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")
        return get_cached_response(
            memoized_functions=memoized_functions,
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
        return get_cached_response(
            memoized_functions=memoized_functions,
            endpoint="/rows",
            dataset=dataset,
            config=config,
            split=split,
        ).send()
