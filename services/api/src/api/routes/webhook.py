# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Any, List, Literal, Optional, TypedDict

from jsonschema import ValidationError, validate
from libcommon.dataset import DatasetError
from libcommon.operations import delete_dataset, update_dataset
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority
from starlette.requests import Request
from starlette.responses import Response

from api.prometheus import StepProfiler
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
    hf_timeout_seconds: Optional[float] = None,
    trust_sender: bool = False,
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
            hf_timeout_seconds=hf_timeout_seconds,
            do_check_support=False,  # always create a job, even if the dataset is not supported
        )
    elif trust_sender:
        # destructive actions (delete, move) require a trusted sender
        if event == "move" and (moved_to := payload["movedTo"]):
            update_dataset(
                dataset=moved_to,
                init_processing_steps=init_processing_steps,
                hf_token=hf_token,
                hf_endpoint=hf_endpoint,
                force=False,
                priority=Priority.NORMAL,
                hf_timeout_seconds=hf_timeout_seconds,
                do_check_support=False,
            )
            delete_dataset(dataset=dataset)
        elif event == "remove":
            delete_dataset(dataset=dataset)


def create_webhook_endpoint(
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_webhook_secret: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> Endpoint:
    async def webhook_endpoint(request: Request) -> Response:
        with StepProfiler(method="webhook_endpoint", step="all"):
            with StepProfiler(method="webhook_endpoint", step="get JSON"):
                try:
                    json = await request.json()
                except Exception:
                    content = {"status": "error", "error": "the body could not be parsed as a JSON"}
                    logging.info("/webhook: the body could not be parsed as a JSON.")
                    return get_response(content, 400)
            logging.info(f"/webhook: {json}")
            with StepProfiler(method="webhook_endpoint", step="parse payload and headers"):
                try:
                    payload = parse_payload(json)
                except ValidationError as e:
                    content = {"status": "error", "error": "the JSON payload is invalid"}
                    logging.info(f"/webhook: the JSON body is invalid. JSON: {json}. Error: {e}")
                    return get_response(content, 400)
                except Exception as e:
                    logging.exception("Unexpected error", exc_info=e)
                    content = {"status": "error", "error": "unexpected error"}
                    logging.warning(f"/webhook: unexpected error while parsing the JSON body is invalid. Error: {e}")
                    return get_response(content, 500)

                HEADER = "x-webhook-secret"
                trust_sender = (
                    hf_webhook_secret is not None
                    and (secret := request.headers.get(HEADER)) is not None
                    and secret == hf_webhook_secret
                )
                if not trust_sender:
                    logging.info(f"/webhook: the sender is not trusted. JSON: {json}")

            with StepProfiler(method="webhook_endpoint", step="process payload"):
                try:
                    process_payload(
                        init_processing_steps=init_processing_steps,
                        payload=payload,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        hf_timeout_seconds=hf_timeout_seconds,
                        trust_sender=trust_sender,
                    )
                except DatasetError as e:
                    content = {"status": "error", "error": "the dataset is not supported"}
                    dataset = payload["repo"]["name"]
                    logging.debug(f"/webhook: the dataset {dataset} is not supported. JSON: {json}. Error: {e}")
                    return get_response(content, 400)
                content = {"status": "ok"}
                return get_response(content, 200)

    return webhook_endpoint
