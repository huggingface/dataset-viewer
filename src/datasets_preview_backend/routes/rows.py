import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.exceptions import (
    Status400Error,
    Status500Error,
    StatusError,
)
from datasets_preview_backend.io.cache import get_rows_response
from datasets_preview_backend.io.queue import is_dataset_in_queue, is_split_in_queue
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def rows_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    split_name = request.query_params.get("split")
    logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")

    try:
        try:
            if (
                not isinstance(dataset_name, str)
                or not isinstance(config_name, str)
                or not isinstance(split_name, str)
            ):
                raise StatusError("Parameters 'dataset', 'config' and 'split' are required", 400)
            rows_response, rows_error, status_code = get_rows_response(dataset_name, config_name, split_name)
            return get_response(rows_response or rows_error, status_code, MAX_AGE_LONG_SECONDS)
        except StatusError as err:
            if err.message == "The split does not exist." and is_dataset_in_queue(dataset_name):
                raise Status400Error("The dataset is being processed. Retry later.", err) from err
            if err.message != "The split cache is empty.":
                raise err
            if is_split_in_queue(dataset_name, config_name, split_name):
                raise Status400Error("The split is being processed. Retry later.", err) from err
            else:
                raise Status500Error("The split cache is empty but no job has been launched.", err) from err
    except StatusError as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)
