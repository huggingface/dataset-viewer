from dataclasses import asdict
from typing import Optional, cast

from datasets import get_dataset_infos

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import InfoDict


def get_info(dataset: str, token: Optional[str] = None) -> InfoDict:
    if not isinstance(dataset, str) and dataset is not None:
        raise TypeError("dataset argument should be a string")
    if dataset is None:
        raise Status400Error("'dataset' is a required query parameter.")
    try:
        dataset_info_dict = get_dataset_infos(dataset, use_auth_token=token)
        info = {}
        for config, dataset_info in dataset_info_dict.items():
            d = asdict(dataset_info)
            if "splits" in d:
                d["splits"] = {split_name: split_info for split_name, split_info in d["splits"].items()}
            info[config] = d
    except FileNotFoundError as err:
        raise Status404Error("The dataset info could not be found.") from err
    except Exception as err:
        raise Status400Error("The dataset info could not be parsed from the dataset.") from err
    return {"dataset": dataset, "info": info}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_info_response(*, dataset: str, token: Optional[str] = None) -> CachedResponse:
    try:
        response = CachedResponse(get_info(dataset, token))
    except (Status400Error, Status404Error) as err:
        response = CachedResponse(err.as_dict(), err.status_code)
    return response


def get_refreshed_info(dataset: str, token: Optional[str] = None) -> InfoDict:
    return cast(InfoDict, get_info_response(dataset, token, _refresh=True)["content"])
