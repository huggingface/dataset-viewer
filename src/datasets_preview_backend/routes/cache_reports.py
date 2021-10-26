import logging
import time
from typing import Any, List, TypedDict, Union

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_SHORT_SECONDS
from datasets_preview_backend.io.mongo import get_dataset_cache
from datasets_preview_backend.models.hf_dataset import HFDataset, get_hf_datasets
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


class CacheReport(TypedDict):
    dataset: str
    tags: List[str]
    downloads: Union[int, None]
    status: str
    error: Union[Any, None]


class CacheReports(TypedDict):
    reports: List[CacheReport]
    created_at: str


# we remove the content because it's too heavy
def get_dataset_report(hf_dataset: HFDataset) -> CacheReport:
    dataset_cache = get_dataset_cache(hf_dataset["id"])
    return {
        "dataset": hf_dataset["id"],
        "tags": hf_dataset["tags"],
        "downloads": hf_dataset["downloads"],
        "status": dataset_cache.status,
        "error": None if dataset_cache.status == "valid" else dataset_cache.content,
    }


def get_cache_reports() -> CacheReports:
    # TODO: cache get_hf_datasets?
    return {
        "reports": [get_dataset_report(hf_dataset) for hf_dataset in get_hf_datasets()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def cache_reports_endpoint(_: Request) -> Response:
    logger.info("/cache-reports")
    return get_response(get_cache_reports(), 200, MAX_AGE_SHORT_SECONDS)
