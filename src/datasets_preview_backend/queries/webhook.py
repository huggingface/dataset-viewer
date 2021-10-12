import logging
from typing import Any, Optional, TypedDict

from datasets_preview_backend.dataset_entries import (
    delete_dataset_entry,
    get_refreshed_dataset_entry,
)
from datasets_preview_backend.exceptions import Status400Error

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


def process_payload(payload: MoonWebhookV2Payload) -> None:
    if payload["add"] is not None:
        get_refreshed_dataset_entry(payload["add"])
    if payload["update"] is not None:
        get_refreshed_dataset_entry(payload["update"])
    if payload["remove"] is not None:
        delete_dataset_entry(payload["remove"])
    return


def post_webhook(json: Any) -> WebHookContent:
    try:
        logger.info(f"webhook: {json}")
        payload = parse_payload(json)
    except Exception as err:
        raise Status400Error("Invalid JSON", err)
    process_payload(payload)
    return {"status": "ok"}
