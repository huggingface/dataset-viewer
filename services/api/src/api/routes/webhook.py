import logging
from typing import Any, Optional, TypedDict

from libcache.cache import create_or_mark_dataset_as_stale, delete_dataset_cache
from libcache.simple_cache import (
    delete_first_rows_responses,
    delete_splits_responses,
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
)
from libqueue.queue import add_dataset_job, add_splits_job
from starlette.requests import Request
from starlette.responses import Response

from api.utils import get_response

logger = logging.getLogger(__name__)


class MoonWebhookV2Payload(TypedDict):
    """
    Payload from a moon-landing webhook call.
    """

    add: Optional[str]
    remove: Optional[str]
    update: Optional[str]


class WebHookContent(TypedDict):
    status: str


def parse_payload(json: Any) -> MoonWebhookV2Payload:
    return {
        "add": str(json["add"]) if "add" in json else None,
        "remove": str(json["remove"]) if "remove" in json else None,
        "update": str(json["update"]) if "update" in json else None,
    }


def get_dataset_name(id: Optional[str]) -> Optional[str]:
    if id is None:
        return None
    dataset_name = id.removeprefix("datasets/")
    if id == dataset_name:
        logger.info(f"ignored because a full dataset id must starts with 'datasets/': {id}")
        return None
    return dataset_name


def try_to_update(id: Optional[str]) -> None:
    dataset_name = get_dataset_name(id)
    if dataset_name is not None:
        logger.debug(f"webhook: refresh {dataset_name}")
        create_or_mark_dataset_as_stale(dataset_name)
        add_dataset_job(dataset_name)
        # new implementation for the /splits endpoint
        mark_splits_responses_as_stale(dataset_name)
        mark_first_rows_responses_as_stale(dataset_name)
        add_splits_job(dataset_name)


def try_to_delete(id: Optional[str]) -> None:
    dataset_name = get_dataset_name(id)
    if dataset_name is not None:
        logger.debug(f"webhook: delete {dataset_name}")
        delete_dataset_cache(dataset_name)
        # new implementation for the /splits endpoint
        delete_splits_responses(dataset_name)
        delete_first_rows_responses(dataset_name)


def process_payload(payload: MoonWebhookV2Payload) -> None:
    try_to_update(payload["add"])
    try_to_update(payload["update"])
    try_to_delete(payload["remove"])


async def webhook_endpoint(request: Request) -> Response:
    try:
        json = await request.json()
    except Exception:
        content = {"status": "error", "error": "the body could not be parsed as a JSON"}
        return get_response(content, 400)
    logger.info(f"/webhook: {json}")
    try:
        payload = parse_payload(json)
    except Exception:
        content = {"status": "error", "error": "the JSON payload is invalid"}
        return get_response(content, 400)

    process_payload(payload)
    content = {"status": "ok"}
    return get_response(content, 200)
