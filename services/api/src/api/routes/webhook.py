# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, Optional, TypedDict

from starlette.requests import Request
from starlette.responses import Response

from api.dataset import delete, is_supported, update
from api.utils import Endpoint, get_response, is_non_empty_string

logger = logging.getLogger(__name__)


class MoonWebhookV1Payload(TypedDict):
    """
    Payload from a moon-landing webhook call.
    """

    add: Optional[str]
    remove: Optional[str]
    update: Optional[str]


class WebHookContent(TypedDict):
    status: str


def parse_payload(json: Any) -> MoonWebhookV1Payload:
    return {
        "add": str(json["add"]) if "add" in json else None,
        "remove": str(json["remove"]) if "remove" in json else None,
        "update": str(json["update"]) if "update" in json else None,
    }


def get_dataset_name(id: Optional[str]) -> Optional[str]:
    if id is None:
        return None
    dataset_name = id.removeprefix("datasets/")
    # temporarily disabled to fix a bug with the webhook
    # (see https://github.com/huggingface/datasets-server/issues/380#issuecomment-1254670923)
    # if id == dataset_name:
    #     logger.info(f"ignored because a full dataset id must starts with 'datasets/': {id}")
    #     return None
    return dataset_name if is_non_empty_string(dataset_name) else None


def process_payload(payload: MoonWebhookV1Payload, hf_endpoint: str, hf_token: Optional[str] = None) -> None:
    unique_datasets = {get_dataset_name(id) for id in {payload["add"], payload["remove"], payload["update"]}}
    for dataset in unique_datasets:
        if dataset is not None:
            if is_supported(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token):
                update(dataset=dataset)
            else:
                delete(dataset=dataset)


def create_webhook_endpoint(hf_endpoint: str, hf_token: Optional[str] = None) -> Endpoint:
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

        process_payload(payload, hf_endpoint, hf_token)
        content = {"status": "ok"}
        return get_response(content, 200)

    return webhook_endpoint
