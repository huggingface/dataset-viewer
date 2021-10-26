# TODO: deprecate?
import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.io.mongo import get_dataset_cache
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def configs_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    logger.info(f"/configs, dataset={dataset_name}")
    dataset_cache = get_dataset_cache(dataset_name=dataset_name)
    if dataset_cache.status != "valid":
        return get_response(dataset_cache.content, dataset_cache.content["status_code"], MAX_AGE_LONG_SECONDS)
    content = {
        "configs": [
            {"dataset": dataset_cache.content["dataset_name"], "config": config["config_name"]}
            for config in dataset_cache.content["configs"]
        ]
    }
    return get_response(content, 200, MAX_AGE_LONG_SECONDS)
