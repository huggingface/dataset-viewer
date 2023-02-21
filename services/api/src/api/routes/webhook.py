# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, List, Literal, Optional, TypedDict

from jsonschema import ValidationError, validate
from libcommon.dataset import DatasetError
from libcommon.operations import delete_dataset, move_dataset, update_dataset
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
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
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["dataset", "model", "space"]},
            },
            "required": ["type", "name"],
        },
    },
    "required": ["event", "repo"],
}


class MoonWebhookV2PayloadRepo(TypedDict):
    type: Literal["model", "dataset", "space"]
    name: str


class MoonWebhookV2Payload(TypedDict):
    """
    Payload from a moon-landing webhook call, v2.
    """

    event: Literal["add", "remove", "update", "move"]
    movedTo: Optional[str]
    repo: MoonWebhookV2PayloadRepo


def parse_payload(json: Any) -> MoonWebhookV2Payload:
    validate(instance=json, schema=schema)
    return json  # type: ignore
    # ^ validate() ensures the content is correct, but does not give the type


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
            priority=Priority.NORMAL,
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
            priority=Priority.NORMAL,
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
        except Exception as e:
            logging.exception("Unexpected error", exc_info=e)
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
