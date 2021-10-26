import logging
from typing import List, TypedDict

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.config import MAX_AGE_LONG_SECONDS
from datasets_preview_backend.models.hf_dataset import get_hf_dataset_names
from datasets_preview_backend.routes._utils import get_response

logger = logging.getLogger(__name__)


class DatasetItem(TypedDict):
    dataset: str


class DatasetsContent(TypedDict):
    datasets: List[DatasetItem]


def get_datasets() -> DatasetsContent:
    return {"datasets": [{"dataset": dataset_name} for dataset_name in get_hf_dataset_names()]}


async def datasets_endpoint(_: Request) -> Response:
    logger.info("/datasets")
    return get_response(get_datasets(), 200, MAX_AGE_LONG_SECONDS)
