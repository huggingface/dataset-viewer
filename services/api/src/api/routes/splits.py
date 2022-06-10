import logging

from libcache.cache import get_splits_response
from libutils.exceptions import Status400Error, StatusError
from starlette.requests import Request
from starlette.responses import Response

from api.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from api.routes._utils import get_response

logger = logging.getLogger(__name__)


async def splits_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    logger.info(f"/splits, dataset={dataset_name}")

    try:
        try:
            if not isinstance(dataset_name, str):
                raise Status400Error("Parameter 'dataset' is required")
            splits_response, splits_error, status_code = get_splits_response(dataset_name)
            return get_response(splits_response or splits_error, status_code, MAX_AGE_LONG_SECONDS)
        except StatusError as err:
            if err.message == "The dataset cache is empty.":
                raise Status400Error("The dataset is being processed. Retry later.", err) from err
            raise err
    except StatusError as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_SHORT_SECONDS)
