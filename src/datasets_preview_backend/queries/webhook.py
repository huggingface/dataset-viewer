import logging
from typing import Any, Optional, TypedDict

from datasets_preview_backend.exceptions import Status400Error, StatusError
from datasets_preview_backend.models.dataset import (
    delete_dataset,
    get_refreshed_dataset,
)

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


def try_to_refresh(id: Optional[str]) -> None:
    dataset_name = get_dataset_name(id)
    if dataset_name is not None:
        logger.debug(f"webhook: refresh {dataset_name}")
        get_refreshed_dataset(dataset_name)


def try_to_delete(id: Optional[str]) -> None:
    dataset_name = get_dataset_name(id)
    if dataset_name is not None:
        logger.debug(f"webhook: delete {dataset_name}")
        delete_dataset(dataset_name)


def process_payload(payload: MoonWebhookV2Payload) -> None:
    try_to_refresh(payload["add"])
    try_to_refresh(payload["update"])
    try_to_delete(payload["remove"])


def post_webhook(json: Any) -> WebHookContent:
    # TODO: respond directly, without waiting for the cache to be refreshed?
    try:
        logger.info(f"webhook: {json}")
        payload = parse_payload(json)
    except Exception as err:
        raise Status400Error("Invalid JSON", err)
    try:
        process_payload(payload)
    except StatusError:
        # the cache has been refreshed, return OK
        pass
    return {"status": "ok"}
