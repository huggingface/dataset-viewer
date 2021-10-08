import logging
from typing import Any

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from datasets_preview_backend.cache_entries import memoized_functions
from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.cache_reports import get_cache_reports
from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.validity_status import get_valid_datasets

logger = logging.getLogger(__name__)


def get_response(func: Any, max_age: int, **kwargs) -> JSONResponse:  # type: ignore
    headers = {"Cache-Control": f"public, max-age={max_age}"}
    try:
        return JSONResponse(func(**kwargs), status_code=200, headers=headers)
    except (Status400Error, Status404Error) as err:
        return JSONResponse(err.as_content(), status_code=err.status_code, headers=headers)


class HealthCheck(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/healthcheck")
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})


class CacheReports(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache-reports")
        return get_response(get_cache_reports, MAX_AGE_SHORT_SECONDS)


class CacheStats(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache")
        return get_response(get_cache_stats, MAX_AGE_SHORT_SECONDS)


class ValidDatasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/valid")
        return get_response(get_valid_datasets, MAX_AGE_SHORT_SECONDS)


class Datasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/datasets")
        return get_response(memoized_functions["/datasets"], MAX_AGE_LONG_SECONDS)


class Infos(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/infos, dataset={dataset}")
        return get_response(
            memoized_functions["/infos"],
            MAX_AGE_LONG_SECONDS,
            dataset=dataset,
            config=config,
        )


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")
        return get_response(memoized_functions["/configs"], MAX_AGE_LONG_SECONDS, dataset=dataset)


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")
        return get_response(
            memoized_functions["/splits"],
            MAX_AGE_LONG_SECONDS,
            dataset=dataset,
            config=config,
        )


class Rows(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        split = request.query_params.get("split")
        logger.info(f"/rows, dataset={dataset}, config={config}, split={split}")
        return get_response(
            memoized_functions["/rows"],
            MAX_AGE_LONG_SECONDS,
            dataset=dataset,
            config=config,
            split=split,
        )
