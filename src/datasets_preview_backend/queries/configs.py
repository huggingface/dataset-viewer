from typing import Optional

from datasets import get_dataset_config_names as get_dataset_config_names

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.responses import SerializedResponse
from datasets_preview_backend.types import ConfigsDict, ResponseJSON


def get_configs(dataset: str, token: Optional[str] = None) -> ConfigsDict:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    try:
        configs = get_dataset_config_names(dataset, use_auth_token=token)
        if len(configs) == 0:
            configs = [DEFAULT_CONFIG_NAME]
    except FileNotFoundError as err:
        raise Status404Error("The dataset could not be found.") from err
    except Exception as err:
        raise Status400Error("The config names could not be parsed from the dataset.") from err

    return {"dataset": dataset, "configs": configs}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_configs_json(dataset: str, token: Optional[str] = None) -> ResponseJSON:
    try:
        response = SerializedResponse(get_configs(dataset, token))
    except (Status400Error, Status404Error) as err:
        response = SerializedResponse(err.as_dict(), err.status_code)
    return response.as_json()
