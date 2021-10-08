import logging

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from datasets_preview_backend.cache_entries import memoized_functions
from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.queries.cache_reports import get_cache_reports
from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.validity_status import get_valid_datasets
from datasets_preview_backend.responses import get_cached_response

logger = logging.getLogger(__name__)


class HealthCheck(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/healthcheck")
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})


class CacheReports(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache-reports")
        return get_cached_response(get_cache_reports, max_age=MAX_AGE_SHORT_SECONDS).send()


class CacheStats(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache")
        return get_cached_response(get_cache_stats, max_age=MAX_AGE_SHORT_SECONDS).send()


class ValidDatasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/valid")
        return get_cached_response(get_valid_datasets, max_age=MAX_AGE_SHORT_SECONDS).send()


class Datasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/datasets")
        return get_cached_response(memoized_functions["/datasets"], max_age=MAX_AGE_LONG_SECONDS).send()


class Infos(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/infos, dataset={dataset}")
        return get_cached_response(
            memoized_functions["/infos"],
            max_age=MAX_AGE_LONG_SECONDS,
            dataset=dataset,
            config=config,
        ).send()


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")
        return get_cached_response(
            memoized_functions["/configs"], max_age=MAX_AGE_LONG_SECONDS, dataset=dataset
        ).send()


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")
        return get_cached_response(
            memoized_functions["/splits"],
            max_age=MAX_AGE_LONG_SECONDS,
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
            memoized_functions["/rows"],
            max_age=MAX_AGE_LONG_SECONDS,
            dataset=dataset,
            config=config,
            split=split,
        ).send()
