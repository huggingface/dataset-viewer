from typing import Optional, cast

from datasets import get_dataset_config_names

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.constants import DATASETS_BLOCKLIST, DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import ConfigsContent


def get_configs(dataset: str, token: Optional[str] = None) -> ConfigsContent:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")
    try:
        configs = get_dataset_config_names(dataset, use_auth_token=token)
        if len(configs) == 0:
            configs = [DEFAULT_CONFIG_NAME]
    except FileNotFoundError as err:
        raise Status404Error("The dataset could not be found.") from err
    except Exception as err:
        raise Status400Error("The config names could not be parsed from the dataset.") from err

    return {"configs": [{"dataset": dataset, "config": d} for d in configs]}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_configs_response(*, dataset: str, token: Optional[str] = None) -> CachedResponse:
    try:
        response = CachedResponse(get_configs(dataset, token))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_content(), err.status_code)
    return response


def get_refreshed_configs(dataset: str, token: Optional[str] = None) -> ConfigsContent:
    return cast(ConfigsContent, get_configs_response(dataset, token, _refresh=True)["content"])
