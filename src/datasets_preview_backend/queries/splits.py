from typing import Optional, Union, cast

from datasets import get_dataset_split_names

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import SplitsDict


def get_splits(dataset: str, config: Union[str, None], token: Optional[str] = None) -> SplitsDict:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    config = DEFAULT_CONFIG_NAME if config is None else config
    if not isinstance(config, str) and config is not None:
        raise TypeError("config argument should be a string")

    try:
        splits = get_dataset_split_names(dataset, config, use_auth_token=token)
    except FileNotFoundError as err:
        raise Status404Error("The dataset config could not be found.") from err
    except ValueError as err:
        if str(err).startswith(f"BuilderConfig {config} not found."):
            raise Status404Error("The dataset config could not be found.") from err
        else:
            raise Status400Error("The split names could not be parsed from the dataset config.") from err
    except Exception as err:
        raise Status400Error("The split names could not be parsed from the dataset config.") from err

    return {"dataset": dataset, "config": config, "splits": splits}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_splits_response(*, dataset: str, config: Union[str, None], token: Optional[str] = None) -> CachedResponse:
    try:
        response = CachedResponse(get_splits(dataset, config, token))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_dict(), err.status_code)
    return response


def get_refreshed_splits(dataset: str, config: Union[str, None], token: Optional[str] = None) -> SplitsDict:
    return cast(SplitsDict, get_splits_response(dataset, config, token, _refresh=True)["content"])
