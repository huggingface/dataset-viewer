import logging
from typing import List, TypedDict, Union

from datasets_preview_backend.constants import DATASETS_BLOCKLIST
from datasets_preview_backend.exceptions import (
    Status400Error,
    StatusError,
    StatusErrorContent,
)
from datasets_preview_backend.io.cache import (  # type: ignore
    CacheNotFoundError,
    cache,
    memoize,
)
from datasets_preview_backend.models.config import Config, get_configs

logger = logging.getLogger(__name__)


class Dataset(TypedDict):
    dataset_name: str
    configs: List[Config]


class DatasetCacheStatus(TypedDict):
    dataset_name: str
    status: str
    content: Union[Dataset, None]
    error: Union[StatusErrorContent, None]


def get_dataset_cache_status(dataset_name: str) -> DatasetCacheStatus:
    try:
        cache_content = get_dataset(dataset_name=dataset_name, _lookup=True)
        status = "valid"
        content = cache_content
        error = None
    except CacheNotFoundError:
        status = "cache_miss"
        content = None
        error = None
    except StatusError as err:
        status = "error"
        content = None
        error = err.as_content()
    except Exception:
        status = "server error"
        content = None
        error = None
    return {
        "dataset_name": dataset_name,
        "status": status,
        "content": content,
        "error": error,
    }


@memoize(cache)  # type:ignore
def get_dataset(*, dataset_name: str) -> Dataset:
    if not isinstance(dataset_name, str) and dataset_name is not None:
        raise TypeError("dataset argument should be a string")
    if dataset_name is None:
        raise Status400Error("'dataset' is a required query parameter.")
    if dataset_name in DATASETS_BLOCKLIST:
        raise Status400Error("this dataset is not supported for now.")

    return {"dataset_name": dataset_name, "configs": get_configs(dataset_name)}


def get_refreshed_dataset(dataset_name: str) -> Dataset:
    return get_dataset(dataset_name=dataset_name, _refresh=True)  # type: ignore


def delete_dataset(dataset_name: str) -> None:
    get_dataset(dataset_name=dataset_name, _delete=True)
