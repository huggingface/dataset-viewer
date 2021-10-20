import base64
import logging
from typing import Any

import orjson
from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.cache_reports import get_cache_reports
from datasets_preview_backend.queries.cache_stats import get_cache_stats
from datasets_preview_backend.queries.configs import get_configs
from datasets_preview_backend.queries.datasets import get_datasets
from datasets_preview_backend.queries.infos import get_infos
from datasets_preview_backend.queries.rows import get_rows
from datasets_preview_backend.queries.splits import get_splits
from datasets_preview_backend.queries.validity_status import get_valid_datasets
from datasets_preview_backend.queries.webhook import post_webhook

logger = logging.getLogger(__name__)


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)


def get_response(func: Any, max_age: int, **kwargs) -> Response:  # type: ignore
    headers = {"Cache-Control": f"public, max-age={max_age}"} if max_age > 0 else {"Cache-Control": "no-store"}
    try:
        return OrjsonResponse(func(**kwargs), status_code=200, headers=headers)
    except (Status400Error, Status404Error) as err:
        return OrjsonResponse(err.as_content(), status_code=err.status_code, headers=headers)


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


class WebHook(HTTPEndpoint):
    async def post(self, request: Request) -> Response:
        logger.info("/webhook")
        headers = {"Cache-Control": "no-store"}
        try:
            json = await request.json()
            return OrjsonResponse(post_webhook(json), status_code=200, headers=headers)
        except (Status400Error, Status404Error) as err:
            return OrjsonResponse(err.as_content(), status_code=err.status_code, headers=headers)

        # return get_response(post_webhook, 0, request=request)


class Datasets(HTTPEndpoint):
    async def get(self, _: Request) -> Response:
        logger.info("/datasets")
        return get_response(get_datasets, MAX_AGE_LONG_SECONDS)


class Infos(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset_name = request.query_params.get("dataset")
        config_name = request.query_params.get("config")
        logger.info(f"/infos, dataset={dataset_name}")
        return get_response(
            get_infos,
            MAX_AGE_LONG_SECONDS,
            dataset_name=dataset_name,
            config_name=config_name,
        )


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset_name = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset_name}")
        return get_response(get_configs, MAX_AGE_LONG_SECONDS, dataset_name=dataset_name)


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset_name = request.query_params.get("dataset")
        config_name = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset_name}, config={config_name}")
        return get_response(
            get_splits,
            MAX_AGE_LONG_SECONDS,
            dataset_name=dataset_name,
            config_name=config_name,
        )


class Rows(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset_name = request.query_params.get("dataset")
        config_name = request.query_params.get("config")
        split_name = request.query_params.get("split")
        logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")
        return get_response(
            get_rows,
            MAX_AGE_LONG_SECONDS,
            dataset_name=dataset_name,
            config_name=config_name,
            split_name=split_name,
        )
