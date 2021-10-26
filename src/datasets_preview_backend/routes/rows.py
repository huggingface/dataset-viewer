import logging

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.exceptions import StatusError
from datasets_preview_backend.io.cache import get_dataset_cache
from datasets_preview_backend.models.config import filter_configs
from datasets_preview_backend.models.split import filter_splits
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


async def rows_endpoint(request: Request) -> Response:
    dataset_name = request.query_params.get("dataset")
    config_name = request.query_params.get("config")
    split_name = request.query_params.get("split")
    logger.info(f"/rows, dataset={dataset_name}, config={config_name}, split={split_name}")
    dataset_cache = get_dataset_cache(dataset_name=dataset_name)
    if dataset_cache.status != "valid":
        return get_response(dataset_cache.content, dataset_cache.content["status_code"], MAX_AGE_LONG_SECONDS)
    if config_name is None:
        # split is ignored if config is not passed
        logger.debug("split argument is ignored since config is not provided")
        split_name = None
    try:
        configs = filter_configs(dataset_cache.content["configs"], config_name)
        content = {
            "columns": [
                {
                    "dataset": dataset_name,
                    "config": config["config_name"],
                    "split": split["split_name"],
                    "column": column,
                }
                for config in configs
                for split in filter_splits(config["splits"], split_name)
                for column in split["columns"]
            ],
            "rows": [
                {
                    "dataset": dataset_name,
                    "config": config["config_name"],
                    "split": split["split_name"],
                    "row": row,
                }
                for config in configs
                for split in filter_splits(config["splits"], split_name)
                for row in split["rows"]
            ],
        }
    except StatusError as err:
        return get_response(err.as_content(), err.status_code, MAX_AGE_LONG_SECONDS)

    return get_response(content, 200, MAX_AGE_LONG_SECONDS)
