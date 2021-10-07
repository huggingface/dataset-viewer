import logging

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from datasets_preview_backend.cache_reports import get_cache_reports
from datasets_preview_backend.exceptions import StatusError
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
        reports = get_cache_reports()
        results = [
            {
                "endpoint": report["endpoint"],
                "kwargs": report["kwargs"],
                "status": report["status"],
                "error": report["content"].as_content() if isinstance(report["content"], StatusError) else None,
            }
            for report in reports
        ]
        return JSONResponse({"reports": results}, headers={"Cache-Control": "no-store"})


class CacheStats(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/cache")
        return JSONResponse(get_cache_stats(), headers={"Cache-Control": "no-store"})


class ValidDatasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/valid")
        return JSONResponse(get_valid_datasets(), headers={"Cache-Control": "no-store"})


class Datasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/datasets")
        return get_cached_response(endpoint="/datasets").send()


class Infos(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/infos, dataset={dataset}")
        return get_cached_response(endpoint="/infos", dataset=dataset, config=config).send()


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")
        return get_cached_response(endpoint="/configs", dataset=dataset).send()


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")
        return get_cached_response(
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
            endpoint="/rows",
            dataset=dataset,
            config=config,
            split=split,
        ).send()
