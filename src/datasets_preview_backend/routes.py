import logging

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.queries.configs import get_configs_json
from datasets_preview_backend.queries.info import get_info_json
from datasets_preview_backend.queries.rows import get_rows_json
from datasets_preview_backend.queries.splits import get_splits_json
from datasets_preview_backend.responses import to_response
from datasets_preview_backend.utils import get_int_value

logger = logging.getLogger(__name__)


class HealthCheck(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        logger.info("/healthcheck")
        return PlainTextResponse("ok", headers={"Cache-Control": "no-store"})


class Info(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/info, dataset={dataset}")

        return to_response(get_info_json(dataset=dataset, token=request.user.token))


class Configs(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        logger.info(f"/configs, dataset={dataset}")

        return to_response(get_configs_json(dataset=dataset, token=request.user.token))


class Splits(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        logger.info(f"/splits, dataset={dataset}, config={config}")

        return to_response(get_splits_json(dataset=dataset, config=config, token=request.user.token))


class Rows(HTTPEndpoint):
    async def get(self, request: Request) -> Response:
        dataset = request.query_params.get("dataset")
        config = request.query_params.get("config")
        split = request.query_params.get("split")
        num_rows = get_int_value(d=request.query_params, key="rows", default=EXTRACT_ROWS_LIMIT)
        logger.info(f"/rows, dataset={dataset}, config={config}, split={split}, num_rows={num_rows}")

        return to_response(
            get_rows_json(dataset=dataset, config=config, split=split, num_rows=num_rows, token=request.user.token)
        )
