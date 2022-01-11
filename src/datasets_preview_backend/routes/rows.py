import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.io.cache import (
    NoError,
    check_dataset,
    get_columns,
    get_error,
    get_rows,
)
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def rows_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    split_name = request.query_params.get("split")
    logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")
    try:
        check_dataset(dataset_name)
        try:
            error = get_error(dataset_name)
            return get_response(error, error["status_code"], MAX_AGE_LONG_SECONDS)
        except NoError:
            columns = get_columns(dataset_name, config_name, split_name)
            rows = get_rows(dataset_name, config_name, split_name)
            if not columns:
                raise Status400Error("No columns found for dataset, config, split.")
            if not rows:
                raise Status400Error("No rows found for dataset, config, split.")
            return get_response({"columns": columns, "rows": rows}, 200, MAX_AGE_LONG_SECONDS)
    except Status400Error as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)
