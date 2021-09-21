from typing import cast

from datasets import list_datasets

from datasets_preview_backend.cache import memoize  # type: ignore
from datasets_preview_backend.config import CACHE_TTL_SECONDS, cache
from datasets_preview_backend.responses import CachedResponse
from datasets_preview_backend.types import DatasetsDict


def get_datasets() -> DatasetsDict:
    # TODO: provide "token: Optional[str] = None" and fetch private datasets as well
    # If an exception is raised, we let starlette generate a 500 error
    datasets = list_datasets(with_community_datasets=True, with_details=False)  # type: ignore
    return {"datasets": datasets}


@memoize(cache, expire=CACHE_TTL_SECONDS)  # type:ignore
def get_datasets_response() -> CachedResponse:
    return CachedResponse(get_datasets())


def get_refreshed_datasets() -> DatasetsDict:
    return cast(DatasetsDict, get_datasets_response(_refresh=True).content)
