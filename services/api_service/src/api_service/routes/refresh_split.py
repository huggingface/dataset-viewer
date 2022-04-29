import logging
from typing import Any, TypedDict

from libqueue.queue import add_split_job
from starlette.requests import Request
from starlette.responses import Response

from api_service.routes._utils import get_response

logger = logging.getLogger(__name__)


class RefreshSplitPayload(TypedDict):
    """
    Payload from a refresh-split call.
    """

    dataset: str
    config: str
    split: str


def extract_key(json: Any, key: str) -> str:
    if key in json and isinstance(json[key], str) and json[key].strip():
        return json[key]
    raise Exception(f"invalid {key}")


def parse_payload(json: Any) -> RefreshSplitPayload:
    return {
        "dataset": extract_key(json, "dataset"),
        "config": extract_key(json, "config"),
        "split": extract_key(json, "split"),
    }


def process_payload(payload: RefreshSplitPayload) -> None:
    dataset_name = payload["dataset"]
    config_name = payload["config"]
    split_name = payload["split"]
    logger.debug(f"webhook: refresh split {dataset_name} - {config_name} - {split_name} ")
    add_split_job(dataset_name, config_name, split_name)


async def refresh_split_endpoint(request: Request) -> Response:
    try:
        json = await request.json()
    except Exception:
        content = {"status": "error", "error": "the body could not be parsed as a JSON"}
        return get_response(content, 400)
    logger.info(f"/refresh-split: {json}")
    try:
        payload = parse_payload(json)
    except Exception:
        content = {"status": "error", "error": "the JSON payload is invalid"}
        return get_response(content, 400)

    process_payload(payload)
    content = {"status": "ok"}
    return get_response(content, 200)
