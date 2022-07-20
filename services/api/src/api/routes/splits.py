import logging

from libcache.cache import get_splits_response, Status400Error, Status500Error, StatusError
from libutils.exceptions import StatusError as StatusErrorBis
from starlette.requests import Request
from starlette.responses import Response

from api.config import MAX_AGE_LONG_SECONDS, MAX_AGE_SHORT_SECONDS
from api.routes._utils import get_response

logger = logging.getLogger(__name__)


async def splits_endpoint(request: Request) -> Response:
    try:
        dataset_name = request.query_params.get("dataset")
        logger.info(f"/splits, dataset={dataset_name}")

        try:
            if not isinstance(dataset_name, str):
                raise Status400Error("Parameter 'dataset' is required")
            splits_response, splits_error, status_code = get_splits_response(dataset_name)
            return get_response(splits_response or splits_error, status_code, MAX_AGE_LONG_SECONDS)
        except (StatusError, StatusErrorBis) as err:
            e = (
                Status400Error("The dataset is being processed. Retry later.")
                if err.message == "The dataset cache is empty."
                else err
            )
            return get_response(e.as_content(), e.status_code, MAX_AGE_SHORT_SECONDS)
    except Exception as err:
        return get_response(Status500Error("Unexpected error.", err).as_content(), 500, MAX_AGE_SHORT_SECONDS)
