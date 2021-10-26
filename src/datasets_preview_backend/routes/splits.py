import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.io.mongo import get_dataset_cache
from datasets_preview_backend.models.config import filter_configs
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def splits_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    logger.info(f"/splits, dataset={dataset_name}, config={config_name}")
    dataset_cache = get_dataset_cache(dataset_name=dataset_name)
    if dataset_cache.status != "valid":
        return get_response(dataset_cache.content, dataset_cache.content["status_code"], MAX_AGE_LONG_SECONDS)
    try:
        content = {
            "splits": [
                {"dataset": dataset_name, "config": config["config_name"], "split": split["split_name"]}
                for config in filter_configs(dataset_cache.content["configs"], config_name)
                for split in config["splits"]
            ]
        }
    except StatusError as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)
    return get_response(content, 200, MAX_AGE_LONG_SECONDS)
