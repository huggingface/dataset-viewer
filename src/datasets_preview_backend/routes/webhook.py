import logging
from typing import Any, Optional, TypedDict

from starlette.requests import Request
from starlette.responses import Response

from datasets_preview_backend.io.cache import create_or_mark_dataset_as_stalled, delete_dataset_cache
from datasets_preview_backend.io.queue import add_dataset_job
from datasets_preview_backend.routes._utils import get_response

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
        create_or_mark_dataset_as_stalled(dataset_name)
        add_dataset_job(dataset_name)


def try_to_delete(id: Optional[str]) -> None:
    dataset_name = get_dataset_name(id)
    if dataset_name is not None:
        logger.debug(f"webhook: delete {dataset_name}")
        delete_dataset_cache(dataset_name)


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
