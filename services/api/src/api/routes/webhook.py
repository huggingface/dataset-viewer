# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, List, Literal, Optional, TypedDict

from jsonschema import ValidationError, validate  # type: ignore
from libcommon.dataset import DatasetError
from libcommon.operations import delete_dataset, move_dataset, update_dataset
from libcommon.processing_graph import ProcessingStep
from starlette.requests import Request
from starlette.responses import Response

from api.utils import Endpoint, get_response

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "event": {"type": "string", "enum": ["add", "remove", "update", "move"]},
        "movedTo": {"type": "string"},
        "repo": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["dataset", "model", "space"]},
                "name": {"type": "string"},
                "gitalyUid": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["type", "name"],
        },
    },
    "required": ["event", "repo"],
}


class MoonWebhookV2PayloadRepo(TypedDict):
    type: Literal["model", "dataset", "space"]
    name: str
    gitalyUid: str
    tags: Optional[List[str]]


class MoonWebhookV2Payload(TypedDict):
    """
    Payload from a moon-landing webhook call, v2.
    """

    event: Literal["add", "remove", "update", "move"]
    movedTo: Optional[str]
    repo: MoonWebhookV2PayloadRepo


def parse_payload(json: Any) -> MoonWebhookV2Payload:
    validate(instance=json, schema=schema)
    return json


def process_payload(
    init_processing_steps: List[ProcessingStep],
    payload: MoonWebhookV2Payload,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> None:
    if payload["repo"]["type"] != "dataset":
        return
    dataset = payload["repo"]["name"]
    if dataset is None:
        return
    event = payload["event"]
    if event in ["add", "update"]:
        update_dataset(
            dataset=dataset,
            init_processing_steps=init_processing_steps,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            force=False,
        )
    elif event == "remove":
        delete_dataset(dataset=dataset)
    elif event == "move":
        moved_to = payload["movedTo"]
        if moved_to is None:
            return
        move_dataset(
            from_dataset=dataset,
            to_dataset=moved_to,
            init_processing_steps=init_processing_steps,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            force=False,
        )


def create_webhook_endpoint(
    init_processing_steps: List[ProcessingStep], hf_endpoint: str, hf_token: Optional[str] = None
) -> Endpoint:
    async def webhook_endpoint(request: Request) -> Response:
        try:
            json = await request.json()
        except Exception:
            content = {"status": "error", "error": "the body could not be parsed as a JSON"}
            return get_response(content, 400)
        logging.info(f"/webhook: {json}")
        try:
            payload = parse_payload(json)
        except ValidationError:
            content = {"status": "error", "error": "the JSON payload is invalid"}
            return get_response(content, 400)
        except Exception:
            content = {"status": "error", "error": "unexpected error"}
            return get_response(content, 500)

        try:
            process_payload(
                init_processing_steps=init_processing_steps,
                payload=payload,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
            )
        except DatasetError:
            content = {"status": "error", "error": "the dataset is not supported"}
            return get_response(content, 400)
        content = {"status": "ok"}
        return get_response(content, 200)

    return webhook_endpoint
