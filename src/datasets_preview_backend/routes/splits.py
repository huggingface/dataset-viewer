import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.exceptions import Status400Error
from datasets_preview_backend.io.cache import (
    NoError,
    check_dataset,
    get_error,
    get_splits,
)
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def splits_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    logger.info(f"/splits, dataset={dataset_name}, config={config_name}")
    try:
        check_dataset(dataset_name)
        try:
            error = get_error(dataset_name)
            return get_response(error, error["status_code"], MAX_AGE_LONG_SECONDS)
        except NoError:
            splits = get_splits(dataset_name, config_name)
            if not splits:
                raise Status400Error("No split found for dataset and config.")
            return get_response({"splits": splits}, 200, MAX_AGE_LONG_SECONDS)
    except Status400Error as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)
